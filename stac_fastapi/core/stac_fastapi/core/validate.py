"""STAC validation module.

Provides validation for STAC items and collections using multiple validation backends:
- Pydantic validation (always enabled)
- Python STAC Validator with FastValidator (fast JSON schema validation)
"""

import asyncio
import json
import logging
import os
import tempfile
import threading

from stac_pydantic import Collection, Item

from stac_fastapi.core.utilities import get_bool_env

logger = logging.getLogger(__name__)

# Suppress verbose logging from stac_validator
logging.getLogger("stac_validator.utilities").setLevel(logging.WARNING)

# Global instances to cache validators and avoid repeated initialization
_fast_validator_instance = None
_validator_lock = threading.Lock()


def _get_fast_validator():
    """Get or create the singleton FastValidator instance.

    Initializes and caches a single FastValidator instance for efficient
    JSON schema validation using fastjsonschema.
    Uses double-checked locking for thread safety.

    Returns:
        The FastValidator instance.

    Raises:
        ImportError: If stac-validator is not installed and ENABLE_STAC_VALIDATOR is true.
    """
    # Only attempt import if validation is enabled
    if not get_bool_env("ENABLE_STAC_VALIDATOR"):
        return None

    global _fast_validator_instance
    if _fast_validator_instance is None:
        with _validator_lock:
            if _fast_validator_instance is None:
                try:
                    from stac_validator.fast_validator import FastValidator

                    _fast_validator_instance = FastValidator
                except ImportError as e:
                    logger.error("stac_validator FastValidator not available")
                    raise ImportError(
                        "STAC validator FastValidator is not installed. "
                        "Install it with: pip install stac-fastapi-core[validator] "
                        "or pip install stac-fastapi-elasticsearch[validator] "
                        "or pip install stac-fastapi-opensearch[validator]"
                    ) from e
    return _fast_validator_instance


def validate_with_fast_validator(
    items: list[dict],
) -> tuple[list[dict], dict[str, list[str]]]:
    """Validate STAC items using FastValidator.

    Uses fastjsonschema for efficient JSON schema validation.
    Validates items as a FeatureCollection for better performance,
    then separates valid items from invalid ones.

    Args:
        items: List of STAC item dictionaries to validate.

    Returns:
        Tuple of (valid_items_list, invalid_items_dict) where invalid_items_dict
        maps error messages to lists of affected item IDs.
    """
    FastValidator = _get_fast_validator()

    if FastValidator is None:
        # Validation disabled
        return items, {}

    valid_items = []
    invalid_items = {}
    errors_by_message: dict[str, list[str]] = {}

    try:
        # Create a FeatureCollection from the items for batch validation
        feature_collection = {
            "type": "FeatureCollection",
            "features": items,
        }

        # Create a temporary file to write the FeatureCollection JSON
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(feature_collection, tmp_file)
            tmp_file_path = tmp_file.name

        try:
            # Validate the entire FeatureCollection using FastValidator
            logger.info(f"Validating FeatureCollection with {len(items)} items")
            validator = FastValidator(tmp_file_path, quiet=True, verbose=False)
            validator.run()

            # Check if there are any errors in the message, even if validator.valid is True
            # (FastValidator might report valid=True for the FeatureCollection structure
            # but have errors for individual items)
            has_errors = False
            if hasattr(validator, "message") and validator.message:
                message_data = validator.message[0]
                errors_list = message_data.get("errors", [])
                has_errors = len(errors_list) > 0

            if validator.valid and not has_errors:
                # All items are valid
                logger.info(f"All {len(items)} items are valid")
                valid_items = items
            else:
                # FeatureCollection validation failed - extract per-item errors
                # FastValidator stores error information in self.message
                logger.warning(
                    "FeatureCollection validation failed, extracting per-item errors"
                )

                # FastValidator stores errors in self.message[0]["errors"]
                if hasattr(validator, "message") and validator.message:
                    message_data = validator.message[0]
                    errors_list = message_data.get("errors", [])

                    # Process error breakdown from FastValidator
                    logger.info(f"Found {len(errors_list)} error types")
                    for error_entry in errors_list:
                        err_msg = error_entry.get(
                            "error_message", "STAC validation failed"
                        )
                        affected_items = error_entry.get("affected_items", [])

                        logger.warning(
                            f"Error: {err_msg} | Affected items: {len(affected_items)}"
                        )

                        # Group by error message
                        if err_msg not in errors_by_message:
                            errors_by_message[err_msg] = []
                        errors_by_message[err_msg].extend(affected_items)

                        for item_id in affected_items:
                            logger.error(
                                f"STAC validation failed for '{item_id}': {err_msg}"
                            )

                    # Identify valid items (those not in any error list)
                    all_invalid_ids = set()
                    for affected_ids in errors_by_message.values():
                        all_invalid_ids.update(affected_ids)

                    valid_items = [
                        item for item in items if item.get("id") not in all_invalid_ids
                    ]
                    logger.info(
                        f"Valid items: {len(valid_items)}, Invalid items: {len(all_invalid_ids)}"
                    )
                else:
                    # Fallback: validate items individually if message not available
                    logger.warning(
                        "FastValidator message not available, validating items individually"
                    )
                    for idx, item in enumerate(items):
                        item_id = item.get("id", f"unknown_id_{idx}")

                        # Create a temporary file for individual item
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json", delete=False
                        ) as item_tmp_file:
                            json.dump(item, item_tmp_file)
                            item_tmp_file_path = item_tmp_file.name

                        try:
                            # Validate individual item
                            item_validator = FastValidator(
                                item_tmp_file_path, quiet=True, verbose=False
                            )
                            item_validator.run()

                            if item_validator.valid:
                                valid_items.append(item)
                            else:
                                # Extract error message from validator.message
                                err_msg = "STAC validation failed"
                                if (
                                    hasattr(item_validator, "message")
                                    and item_validator.message
                                ):
                                    msg_data = item_validator.message[0]
                                    errors_list = msg_data.get("errors", [])
                                    if errors_list:
                                        err_msg = errors_list[0].get(
                                            "error_message", "STAC validation failed"
                                        )

                                # Group by error message
                                if err_msg not in errors_by_message:
                                    errors_by_message[err_msg] = []
                                errors_by_message[err_msg].append(item_id)
                                logger.error(
                                    f"STAC validation failed for '{item_id}': {err_msg}"
                                )
                        finally:
                            # Clean up temporary file
                            try:
                                os.unlink(item_tmp_file_path)
                            except OSError:
                                pass
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass

        invalid_items = errors_by_message

    except Exception as exc:
        logger.error(f"Batch validation request failed: {exc}", exc_info=True)
        error_msg = f"Batch validation failed: {str(exc)}"
        item_ids = [
            item.get("id", f"unknown_id_{idx}") for idx, item in enumerate(items)
        ]
        invalid_items[error_msg] = item_ids

    logger.info(
        f"Validation complete: {len(valid_items)} valid, {sum(len(v) for v in invalid_items.values())} invalid"
    )
    return valid_items, invalid_items


def validate_stac(
    stac_data: dict | Item | Collection,
    pydantic_model: type[Item] | type[Collection] = Item,
) -> Item | Collection:
    """Validate a single STAC item or collection using optional STAC validator.

    If stac_data is already a Pydantic model object, Pydantic validation is skipped
    (assuming it was already validated by FastAPI). Only STAC validator is run if enabled.

    Args:
        stac_data: STAC data as dict or Pydantic model object.
        pydantic_model: The Pydantic model class to use for validation (Item or Collection).

    Returns:
        Validated STAC object (Item or Collection).

    Raises:
        ValueError: If STAC validation fails.
    """
    # 1. Pydantic Parsing/Validation
    # If already a Pydantic model object, skip Pydantic validation (FastAPI already validated it)
    if isinstance(stac_data, (Item, Collection)):
        stac_obj = stac_data
        stac_dict = stac_data.model_dump(mode="json")
    else:
        # For dict input, validate with Pydantic first
        stac_obj = pydantic_model(**stac_data)
        stac_dict = stac_data

    # 2. STAC Validator (optional, enabled via ENABLE_STAC_VALIDATOR env var)
    if get_bool_env("ENABLE_STAC_VALIDATOR"):
        FastValidator = _get_fast_validator()
        item_id = stac_dict.get("id", "unknown_id")

        # Create a temporary file to write the item JSON
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(stac_dict, tmp_file)
            tmp_file_path = tmp_file.name

        try:
            # Validate using FastValidator
            validator = FastValidator(tmp_file_path, quiet=True, verbose=False)
            validator.run()

            if not validator.valid:
                err_msg = "STAC validation failed"
                if hasattr(validator, "message") and validator.message:
                    err_msg = validator.message
                elif hasattr(validator, "errors") and validator.errors:
                    err_msg = "; ".join(
                        str(e) if isinstance(e, str) else str(e)
                        for e in validator.errors
                    )
                raise ValueError(f"STAC validation failed for '{item_id}': {err_msg}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass

    return stac_obj


async def async_validate_stac(
    stac_data: dict | Item | Collection,
    pydantic_model: type[Item] | type[Collection] = Item,
) -> Item | Collection:
    """Asynchronous wrapper for validate_stac.

    Offloads the CPU-bound STAC validation to a separate thread to prevent
    blocking the FastAPI asyncio event loop during API requests.

    Args:
        stac_data: STAC data as dict or Pydantic model.
        pydantic_model: The Pydantic model class to use for validation (Item or Collection).

    Returns:
        Validated STAC object (Item or Collection).

    Raises:
        ValueError: If validation fails.
    """
    return await asyncio.to_thread(validate_stac, stac_data, pydantic_model)


async def async_validate_with_fast_validator(
    items: list[dict],
) -> tuple[list[dict], dict[str, list[str]]]:
    """Asynchronously validate STAC items using FastValidator.

    Offloads the CPU-bound validation to a separate thread to prevent
    blocking the FastAPI asyncio event loop.

    Args:
        items: List of STAC item dictionaries to validate.

    Returns:
        Tuple of (valid_items_list, invalid_items_dict) where invalid_items_dict
        maps error messages to lists of affected item IDs.
    """
    return await asyncio.to_thread(validate_with_fast_validator, items)
