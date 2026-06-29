"""FastAPI application."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Type

from fastapi import FastAPI

from stac_fastapi.api.app import StacApi
from stac_fastapi.api.models import (
    ItemCollectionUri,
    create_get_request_model,
    create_post_request_model,
    create_request_model,
)
from stac_fastapi.core.core import CoreClient
from stac_fastapi.core.exceptions import QueuedSuccess, queued_success_handler
from stac_fastapi.core.extensions import QueryExtension
from stac_fastapi.core.extensions.collections_search import (
    CollectionsSearchEndpointExtension,
)
from stac_fastapi.core.extensions.fields import FieldsExtension
from stac_fastapi.core.rate_limit import setup_rate_limit
from stac_fastapi.core.route_dependencies import get_route_dependencies
from stac_fastapi.core.utilities import get_bool_env
from stac_fastapi.extensions.core import FreeTextExtension, SortExtension
from stac_fastapi.extensions.core.fields import FieldsConformanceClasses
from stac_fastapi.extensions.core.free_text import FreeTextConformanceClasses
from stac_fastapi.extensions.core.query import QueryConformanceClasses
from stac_fastapi.extensions.core.sort import SortConformanceClasses
from stac_fastapi.opensearch.config import OpensearchSettings
from stac_fastapi.opensearch.database_logic import (
    create_collection_index,
    create_index_templates,
)
from stac_fastapi.sfeos_helpers.database.utils import sentry_initialize

sentry_enable = get_bool_env("SENTRY_ENABLE", default=False)

if sentry_enable:
    sentry_initialize(
        dsn=os.getenv("SENTRY_DSN"),
        environment=os.getenv("SENTRY_ENVIRONMENT", "staging"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        ca_certs=os.getenv("SENTRY_CA_CERTS", None),
    )

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def instantiate_api(
    settings: OpensearchSettings | None = None,
    client: Type[CoreClient] = CoreClient,
    extensions_config=None,
) -> StacApi:
    """Instantiate the STAC API.

    Args:
        settings: The application settings, must be an instance of `OpensearchSettings`.
        client: The client class to use for the API, must be a subclass of `CoreClient`.
        extensions_config: The extensions configuration object.
    Returns:
        An instance of the STAC API.
    """
    settings = settings or OpensearchSettings()

    if extensions_config is None:
        from stac_fastapi.opensearch.models import Extensions

        extensions_config = Extensions(settings=settings)

    database_logic = extensions_config.database_logic
    session = extensions_config.session

    search_extensions = extensions_config.search
    aggregation_extension = extensions_config.aggregation

    extensions = [aggregation_extension] + search_extensions

    # Collection search related variables
    collections_get_request_model = None
    collection_search_post_request_model = None
    collection_search_ext = extensions_config.collection_search

    if settings.enable_collections_search or settings.enable_collections_search_route:
        if collection_search_ext:
            collections_get_request_model = collection_search_ext.GET

    if settings.enable_collections_search:
        from stac_fastapi.extensions.core import CollectionSearchPostExtension

        if collection_search_ext:
            collection_search_post_request_model = create_post_request_model(
                [
                    QueryExtension(
                        conformance_classes=[QueryConformanceClasses.COLLECTIONS]
                    ),
                    SortExtension(
                        conformance_classes=[SortConformanceClasses.COLLECTIONS]
                    ),
                    FieldsExtension(
                        conformance_classes=[FieldsConformanceClasses.COLLECTIONS]
                    ),
                ]
            )

            collection_search_post_ext = CollectionSearchPostExtension(
                client=client(
                    database=database_logic,
                    session=session,
                    post_request_model=collection_search_post_request_model,
                    landing_page_id=os.getenv(
                        "STAC_FASTAPI_LANDING_PAGE_ID", "stac-fastapi"
                    ),
                ),
                settings=settings,
                POST=collection_search_post_request_model,
                conformance_classes=[
                    "https://api.stacspec.org/v1.0.0-rc.1/collection-search",
                    QueryConformanceClasses.COLLECTIONS,
                ],
            )
            extensions.append(collection_search_ext)
            extensions.append(collection_search_post_ext)

    if settings.enable_collections_search_route:
        collection_search_ext = extensions_config.collection_search
        if collection_search_ext:
            if not collection_search_post_request_model:
                collection_search_post_request_model = create_post_request_model(
                    [
                        QueryExtension(
                            conformance_classes=[QueryConformanceClasses.COLLECTIONS]
                        ),
                        SortExtension(
                            conformance_classes=[SortConformanceClasses.COLLECTIONS]
                        ),
                        FieldsExtension(
                            conformance_classes=[FieldsConformanceClasses.COLLECTIONS]
                        ),
                    ]
                )

            collections_search_endpoint_ext = CollectionsSearchEndpointExtension(
                client=client(
                    database=database_logic,
                    session=session,
                    post_request_model=collection_search_post_request_model,
                    landing_page_id=os.getenv(
                        "STAC_FASTAPI_LANDING_PAGE_ID", "stac-fastapi"
                    ),
                ),
                settings=settings,
                GET=collections_get_request_model,
                POST=collection_search_post_request_model,
                conformance_classes=[
                    "https://api.stacspec.org/v1.0.0-rc.1/collection-search",
                    QueryConformanceClasses.COLLECTIONS,
                ],
            )
            extensions.append(collection_search_ext)
            extensions.append(collections_search_endpoint_ext)

    extensions.extend(extensions_config.catalogs)

    database_logic.extensions = [type(ext).__name__ for ext in extensions]

    post_request_model = create_post_request_model(search_extensions)

    items_get_request_model = create_request_model(
        model_name="ItemCollectionUri",
        base_model=ItemCollectionUri,
        extensions=[
            SortExtension(
                conformance_classes=[SortConformanceClasses.ITEMS],
            ),
            QueryExtension(
                conformance_classes=[QueryConformanceClasses.ITEMS],
            ),
            extensions_config.filter,
            FieldsExtension(conformance_classes=[FieldsConformanceClasses.ITEMS]),
            FreeTextExtension(
                conformance_classes=[FreeTextConformanceClasses.ITEMS],
            ),
        ],
        request_type="GET",
    )

    app_config = {
        "title": os.getenv("STAC_FASTAPI_TITLE", "stac-fastapi-opensearch"),
        "description": os.getenv("STAC_FASTAPI_DESCRIPTION", "stac-fastapi-opensearch"),
        "api_version": os.getenv("STAC_FASTAPI_VERSION", "6.17.2"),
        "settings": settings,
        "extensions": extensions,
        "client": client(
            database=database_logic,
            session=session,
            post_request_model=post_request_model,
            landing_page_id=os.getenv("STAC_FASTAPI_LANDING_PAGE_ID", "stac-fastapi"),
        ),
        "search_get_request_model": create_get_request_model(search_extensions),
        "search_post_request_model": post_request_model,
        "items_get_request_model": items_get_request_model,
        "route_dependencies": get_route_dependencies(),
    }

    if collections_get_request_model:
        app_config["collections_get_request_model"] = collections_get_request_model

    api = StacApi(**app_config)
    return api


_api = None
_app = None


def get_api():
    """Get or create the API instance."""
    global _api
    if _api is None:
        _api = instantiate_api()
    return _api


def get_app():
    """Get or create the FastAPI app instance."""
    global _app
    if _app is None:
        api = get_api()
        _app = api.app

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Lifespan handler for FastAPI app. Initializes index templates and collections at startup."""
            await create_index_templates()
            await create_collection_index()
            yield

        _app.router.lifespan_context = lifespan

        # Register custom exception handler for queued items (202 Accepted)
        _app.add_exception_handler(QueuedSuccess, queued_success_handler)
        _app.root_path = os.getenv("STAC_FASTAPI_ROOT_PATH", "")

    return _app


# Lazy app initialization - will be created on first access
class AppProxy:
    """Lazy proxy for the FastAPI app that initializes on first access."""

    _app = None
    _initialized = False

    def _ensure_initialized(self):
        """Ensure the app is initialized."""
        if self._app is None:
            self._app = get_app()
            if not self._initialized:
                # Setup metrics and rate limiting
                try:
                    from stac_fastapi.sfeos_helpers.metrics import get_instrumentator

                    metrics = get_instrumentator()
                    metrics.instrument(self._app).expose(self._app, endpoint="/metrics")
                except ImportError:
                    logger.warning(
                        "prometheus-fastapi-instrumentator not installed; metrics endpoint disabled"
                    )
                setup_rate_limit(
                    self._app, rate_limit=os.getenv("STAC_FASTAPI_RATE_LIMIT")
                )
                self._initialized = True

    def __getattr__(self, name):
        """Get attribute from the underlying app."""
        self._ensure_initialized()
        return getattr(self._app, name)

    def __call__(self, *args, **kwargs):
        """Call the underlying app."""
        self._ensure_initialized()
        return self._app(*args, **kwargs)

    def __iter__(self):
        """Return iterator of the underlying app."""
        self._ensure_initialized()
        return iter(self._app)


app = AppProxy()


def run() -> None:
    """Run app from command line using uvicorn if available."""
    try:
        import uvicorn

        settings = OpensearchSettings()

        uvicorn.run(
            "stac_fastapi.opensearch.app:app",
            host=settings.app_host,
            port=settings.app_port,
            log_level="info",
            reload=settings.reload,
        )
    except ImportError:
        raise RuntimeError("Uvicorn must be installed in order to use command")


if __name__ == "__main__":
    run()


def create_handler(app):
    """Create a handler to use with AWS Lambda if mangum available."""
    try:
        from mangum import Mangum

        return Mangum(app)
    except ImportError:
        return None


handler = create_handler(app)
