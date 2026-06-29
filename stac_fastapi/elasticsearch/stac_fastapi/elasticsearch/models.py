"""Models for the Elasticsearch backend."""
import logging
from dataclasses import dataclass, field

from stac_fastapi.core.core import BulkTransactionsClient, TransactionsClient
from stac_fastapi.core.extensions import QueryExtension
from stac_fastapi.core.extensions.aggregation import (
    EsAggregationExtensionGetRequest,
    EsAggregationExtensionPostRequest,
)
from stac_fastapi.core.extensions.fields import FieldsExtension
from stac_fastapi.core.session import Session
from stac_fastapi.elasticsearch.config import ElasticsearchSettings
from stac_fastapi.elasticsearch.database_logic import DatabaseLogic
from stac_fastapi.extensions.core import (
    AggregationExtension,
    CollectionSearchExtension,
    CollectionSearchFilterExtension,
    CollectionSearchPostExtension,
    FilterExtension,
    FreeTextExtension,
    SortExtension,
    TokenPaginationExtension,
    TransactionExtension,
)
from stac_fastapi.extensions.core.fields import FieldsConformanceClasses
from stac_fastapi.extensions.core.filter import FilterConformanceClasses
from stac_fastapi.extensions.core.free_text import FreeTextConformanceClasses
from stac_fastapi.extensions.core.query import QueryConformanceClasses
from stac_fastapi.extensions.core.sort import SortConformanceClasses
from stac_fastapi.extensions.third_party import BulkTransactionExtension
from stac_fastapi.sfeos_helpers.aggregation import EsAsyncBaseAggregationClient
from stac_fastapi.sfeos_helpers.filter import EsAsyncBaseFiltersClient
from stac_fastapi.types.extension import ApiExtension

logger = logging.getLogger(__name__)


@dataclass
class Extensions:
    """Elasticsearch extensions configuration."""

    settings: ElasticsearchSettings = field(default_factory=ElasticsearchSettings)
    database_logic: DatabaseLogic = field(default_factory=DatabaseLogic)
    session: Session = field(default=None)

    def __post_init__(self):
        """Initialize session with the configured settings."""
        if self.session is None:
            self.session = Session.create_from_settings(self.settings)

    def _get_filter_extension(self) -> FilterExtension:
        """Create filter extension."""
        extension = FilterExtension(
            client=EsAsyncBaseFiltersClient(
                database=self.database_logic, settings=self.settings
            )
        )
        extension.conformance_classes.append(
            FilterConformanceClasses.ADVANCED_COMPARISON_OPERATORS
        )
        return extension

    def _get_aggregation_extension(self) -> AggregationExtension:
        """Create aggregation extension."""
        extension = AggregationExtension(
            client=EsAsyncBaseAggregationClient(
                database=self.database_logic,
                session=self.session,
                settings=self.settings,
            )
        )
        extension.POST = EsAggregationExtensionPostRequest
        extension.GET = EsAggregationExtensionGetRequest
        return extension

    def _get_fields_extension(self) -> FieldsExtension:
        """Create fields extension."""
        extension = FieldsExtension()
        extension.conformance_classes.append(FieldsConformanceClasses.ITEMS)
        return extension

    @property
    def search(self) -> list[ApiExtension]:
        """Get search extensions."""
        extensions = [
            self._get_fields_extension(),
            QueryExtension(),
            SortExtension(),
            TokenPaginationExtension(),
            self._get_filter_extension(),
            FreeTextExtension(
                conformance_classes=[FreeTextConformanceClasses.SEARCH],
            ),
        ]

        if self.settings.enable_transactions_extensions:
            extensions.insert(
                0,
                TransactionExtension(
                    client=TransactionsClient(
                        database=self.database_logic,
                        session=self.session,
                        settings=self.settings,
                    ),
                    settings=self.settings,
                ),
            )
            extensions.insert(
                1,
                BulkTransactionExtension(
                    client=BulkTransactionsClient(
                        database=self.database_logic,
                        session=self.session,
                        settings=self.settings,
                    )
                ),
            )

        return extensions

    @property
    def aggregation(self) -> AggregationExtension:
        """Get aggregation extension."""
        return self._get_aggregation_extension()

    @property
    def filter(self) -> FilterExtension:
        """Get filter extension."""
        return self._get_filter_extension()

    @property
    def collection_search(self) -> CollectionSearchExtension | None:
        """Get collection search extension."""
        if (
            self.settings.enable_collections_search
            or self.settings.enable_collections_search_route
        ):
            collection_search_extensions = [
                QueryExtension(
                    conformance_classes=[QueryConformanceClasses.COLLECTIONS]
                ),
                SortExtension(conformance_classes=[SortConformanceClasses.COLLECTIONS]),
                FieldsExtension(
                    conformance_classes=[FieldsConformanceClasses.COLLECTIONS]
                ),
                CollectionSearchFilterExtension(
                    conformance_classes=[FilterConformanceClasses.COLLECTIONS]
                ),
                FreeTextExtension(
                    conformance_classes=[FreeTextConformanceClasses.COLLECTIONS]
                ),
            ]
            return CollectionSearchExtension.from_extensions(
                collection_search_extensions
            )
        return None

    @property
    def collection_search_post(self) -> CollectionSearchPostExtension | None:
        """Get collection search POST extension."""
        if self.settings.enable_collections_search:
            return CollectionSearchPostExtension(
                client=None,  # Will be set in app.py
                settings=self.settings,
                POST=None,  # Will be set in app.py
                conformance_classes=[
                    "https://api.stacspec.org/v1.0.0-rc.1/collection-search",
                    QueryConformanceClasses.COLLECTIONS,
                    FilterConformanceClasses.COLLECTIONS,
                    FreeTextConformanceClasses.COLLECTIONS,
                    SortConformanceClasses.COLLECTIONS,
                    FieldsConformanceClasses.COLLECTIONS,
                ],
            )
        return None

    @property
    def catalogs(self) -> list[ApiExtension]:
        """Get catalogs extensions."""
        logger.info(
            "ENABLE_CATALOGS_ROUTE is set to %s",
            self.settings.enable_catalogs_route,
        )
        logger.info(
            "HIDE_ALTERNATE_PARENTS is set to %s",
            self.settings.hide_alternate_parents,
        )

        if self.settings.enable_catalogs_route:
            try:
                from stac_fastapi_catalogs_extension import (
                    CatalogsExtension,
                    CatalogsTransactionExtension,
                )

                from stac_fastapi.core.catalogs_client import CatalogsClient

                catalogs_client = CatalogsClient(database=self.database_logic)

                catalogs_extension = CatalogsExtension(
                    client=catalogs_client,
                    settings=self.settings.model_dump(),
                    hide_alternate_parents=self.settings.hide_alternate_parents,
                )
                extensions = [catalogs_extension]
                logger.info("CatalogsExtension (read-only) enabled successfully.")

                if self.settings.enable_transactions_extensions:
                    catalogs_transaction_extension = CatalogsTransactionExtension(
                        client=catalogs_client,
                        settings=self.settings.model_dump(),
                    )
                    extensions.append(catalogs_transaction_extension)
                    logger.info("CatalogsTransactionExtension enabled successfully.")

                return extensions
            except ImportError as e:
                logger.warning(
                    "ENABLE_CATALOGS_ROUTE is set to true, but the catalogs extension is not installed. "
                    "Please install it with: pip install stac-fastapi-core[catalogs]. "
                    f"Error: {e}"
                )
                return []
        return []
