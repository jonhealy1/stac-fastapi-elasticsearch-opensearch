import copy
import json
import os
from typing import Any, Callable, List

import pytest
import pytest_asyncio
from fastapi import Depends, HTTPException
from fastapi import params as fastapi_params
from fastapi import security, status
from fastapi.routing import BaseRoute
from httpx import ASGITransport, AsyncClient
from pydantic import ConfigDict
from stac_pydantic import api

from stac_fastapi.core.basic_auth import BasicAuth
from stac_fastapi.core.core import (
    BulkTransactionsClient,
    CoreClient,
    TransactionsClient,
)
from stac_fastapi.core.rate_limit import setup_rate_limit
from stac_fastapi.sfeos_helpers.mappings import ITEMS_INDEX_PREFIX
from stac_fastapi.types.config import Settings

os.environ.setdefault("ENABLE_COLLECTIONS_SEARCH_ROUTE", "true")
os.environ.setdefault("ENABLE_CATALOGS_ROUTE", "false")
os.environ.setdefault("DATABASE_REFRESH", "true")

if os.getenv("BACKEND", "elasticsearch").lower() == "opensearch":
    from stac_fastapi.opensearch.app import instantiate_api
    from stac_fastapi.opensearch.config import AsyncOpensearchSettings as AsyncSettings
    from stac_fastapi.opensearch.config import OpensearchSettings as SearchSettings
    from stac_fastapi.opensearch.database_logic import (
        DatabaseLogic,
        create_collection_index,
        create_index_templates,
    )
else:
    from stac_fastapi.elasticsearch.app import instantiate_api
    from stac_fastapi.elasticsearch.config import (
        AsyncElasticsearchSettings as AsyncSettings,
    )
    from stac_fastapi.elasticsearch.config import (
        ElasticsearchSettings as SearchSettings,
    )
    from stac_fastapi.elasticsearch.database_logic import (
        DatabaseLogic,
        create_collection_index,
        create_index_templates,
    )


_app_config_cache = None


def get_app_config():
    """Get the app configuration dict from instantiate_api()."""
    global _app_config_cache
    if _app_config_cache is None:
        api = instantiate_api()
        _app_config_cache = {
            "app": api.app,
            "title": api.app.title,
            "description": api.app.description,
            "api_version": api.app.version,
            "settings": api.settings,
            "extensions": api.extensions,
            "client": api.client,
            "search_get_request_model": api.search_get_request_model,
            "search_post_request_model": api.search_post_request_model,
            "items_get_request_model": api.items_get_request_model,
            "collections_get_request_model": api.collections_get_request_model,
            "route_dependencies": api.route_dependencies,
        }
    return _app_config_cache


# Lazy initialization - will be populated on first access
class AppConfigProxy(dict):
    """Lazy proxy for app_config that initializes on first access."""

    def __getitem__(self, key):
        return get_app_config()[key]

    def __setitem__(self, key, value):
        get_app_config()[key] = value

    def __contains__(self, key):
        return key in get_app_config()

    def __iter__(self):
        return iter(get_app_config())

    def keys(self):
        return get_app_config().keys()

    def values(self):
        return get_app_config().values()

    def items(self):
        return get_app_config().items()

    def get(self, key, default=None):
        return get_app_config().get(key, default)

    def copy(self):
        return get_app_config().copy()


app_config = AppConfigProxy()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "datetime_filtering: matches datetime_filtering mark"
    )
    config.addinivalue_line(
        "filterwarnings", "ignore:Duplicate Operation ID:UserWarning"
    )


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class Context:
    def __init__(self, item, collection):
        self.item = item
        self.collection = collection


class MockRequest:
    base_url = "http://test-server"
    url = "http://test-server/test"
    headers = {}
    query_params = {}

    def __init__(
        self,
        method: str = "GET",
        url: str = "XXXX",
        app: Any | None = None,
        query_params: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
    ):
        self.method = method
        self.url = url
        self.app = app
        self.query_params = query_params or {"limit": "10"}
        self.headers = headers or {"content-type": "application/json"}


class TestSettings(AsyncSettings):
    model_config = ConfigDict(env_file=".env.test")


settings = TestSettings()
Settings.set(settings)


def _load_file(filename: str) -> dict:
    with open(os.path.join(DATA_DIR, filename)) as file:
        return json.load(file)


_test_item_prototype = _load_file("test_item.json")
_test_collection_prototype = _load_file("test_collection.json")


@pytest.fixture
def load_test_data() -> Callable[[str], dict]:
    return _load_file


@pytest.fixture
def test_item() -> dict:
    return copy.deepcopy(_test_item_prototype)


@pytest.fixture
def test_collection() -> dict:
    return copy.deepcopy(_test_collection_prototype)


async def create_collection(txn_client: TransactionsClient, collection: dict) -> None:
    await txn_client.create_collection(
        api.Collection(**dict(collection)), request=MockRequest, refresh=True
    )


async def create_item(txn_client: TransactionsClient, item: dict) -> None:
    if "collection" in item:
        await txn_client.create_item(
            collection_id=item["collection"],
            item=api.Item(**item),
            request=MockRequest,
            refresh=True,
        )
    else:
        await txn_client.create_item(
            collection_id=item["features"][0]["collection"],
            item=api.ItemCollection(**item),
            request=MockRequest,
            refresh=True,
        )


async def delete_collections_and_items(txn_client: TransactionsClient) -> None:
    await refresh_indices(txn_client)
    await txn_client.database.delete_items()
    await txn_client.database.delete_collections()
    await txn_client.database.client.indices.delete(index=f"{ITEMS_INDEX_PREFIX}*")
    await txn_client.database.async_index_selector.refresh_cache()


async def refresh_indices(txn_client: TransactionsClient) -> None:
    try:
        await txn_client.database.client.indices.refresh(index="_all")
    except Exception:
        pass


@pytest_asyncio.fixture()
async def ctx(txn_client: TransactionsClient, test_collection, test_item):
    # todo remove one of these when all methods use it
    await delete_collections_and_items(txn_client)

    await create_collection(txn_client, test_collection)
    await create_item(txn_client, test_item)

    yield Context(item=test_item, collection=test_collection)

    await delete_collections_and_items(txn_client)


database = DatabaseLogic()
settings = SearchSettings()


@pytest.fixture
def core_client():
    return CoreClient(database=database, session=None)


@pytest.fixture
def txn_client():
    return TransactionsClient(database=database, session=None, settings=settings)


@pytest.fixture
def bulk_txn_client():
    return BulkTransactionsClient(database=database, session=None, settings=settings)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def app():
    return app_config["app"]


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def app_client(app):
    await create_index_templates()
    await create_collection_index()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test-server"
    ) as c:
        yield c


@pytest_asyncio.fixture()
async def app_rate_limit():
    """Fixture to get the FastAPI app with test-specific rate limiting."""
    if os.getenv("BACKEND", "elasticsearch").lower() == "opensearch":
        from stac_fastapi.opensearch.app import instantiate_api as instantiate_api_local
    else:
        from stac_fastapi.elasticsearch.app import (
            instantiate_api as instantiate_api_local,
        )

    api = instantiate_api_local()
    app = api.app
    setup_rate_limit(app, rate_limit="2/minute")

    return app


@pytest_asyncio.fixture()
async def app_client_rate_limit(app_rate_limit):
    await create_index_templates()
    await create_collection_index()

    async with AsyncClient(
        transport=ASGITransport(app=app_rate_limit), base_url="http://test-server"
    ) as c:
        yield c


@pytest_asyncio.fixture()
async def app_basic_auth():
    """Fixture to get the FastAPI app with basic auth configured."""
    if os.getenv("BACKEND", "elasticsearch").lower() == "opensearch":
        from stac_fastapi.opensearch.app import instantiate_api as instantiate_api_local
    else:
        from stac_fastapi.elasticsearch.app import (
            instantiate_api as instantiate_api_local,
        )

    api = instantiate_api_local()
    app = api.app

    # 2. CRITICAL FIX: Rebuild extensions and clients from scratch!
    # This ensures app_basic_auth gets its own fresh APIRouters.
    # If we share the global extensions, our monkey-patch will poison
    # the routes for the entire test suite (FastAPI >= 0.137 shared state leak).
    auth_settings = AsyncSettings()
    aggregation_extension = AggregationExtension(
        client=EsAsyncBaseAggregationClient(
            database=database, session=None, settings=auth_settings
        )
    )
    aggregation_extension.POST = EsAggregationExtensionPostRequest
    aggregation_extension.GET = EsAggregationExtensionGetRequest

    auth_extensions = [
        aggregation_extension,
        FieldsExtension(),
        SortExtension(),
        QueryExtension(),
        TokenPaginationExtension(),
        FilterExtension(),
        FreeTextExtension(),
        TransactionExtension(
            client=TransactionsClient(
                database=database, session=None, settings=auth_settings
            ),
            settings=auth_settings,
        ),
    ]
    test_config["extensions"] = auth_extensions
    test_config["client"] = CoreClient(
        database=database,
        session=None,
        extensions=auth_extensions,
        post_request_model=test_config["search_post_request_model"],
    )

    # 3. Create basic auth dependency
    basic_auth = Depends(
        BasicAuth(credentials=[{"username": "admin", "password": "admin"}])
    )

    # 4. Define public routes that don't require auth
    public_paths = {
        "/": ["GET"],
        "/conformance": ["GET"],
        "/collections/{collection_id}/items/{item_id}": ["GET"],
        "/search": ["GET", "POST"],
        "/collections": ["GET"],
        "/collections/{collection_id}": ["GET"],
        "/collections/{collection_id}/items": ["GET"],
        "/queryables": ["GET"],
        "/collections/{collection_id}/queryables": ["GET"],
        "/_mgmt/ping": ["GET"],
    }

    # Initialize route dependencies with public paths
    route_dependencies = [
        (
            [{"path": path, "method": method} for method in methods],
            [],  # No auth for public routes
        )
        for path, methods in public_paths.items()
    ]

    # Add catch-all route with basic auth
    route_dependencies.extend(
        [
            (
                [{"path": "*", "method": "*"}],
                [basic_auth],
            )  # Require auth for all other routes
        ]
    )

    # Note: Route dependencies are already configured in the app via instantiate_api
    # We cannot add middleware after the app has started, so return as-is
    return app


@pytest_asyncio.fixture()
async def app_client_basic_auth(app_basic_auth):
    await create_index_templates()
    await create_collection_index()

    async with AsyncClient(
        transport=ASGITransport(app=app_basic_auth), base_url="http://test-server"
    ) as c:
        yield c


def must_be_bob(
    credentials: security.HTTPBasicCredentials = Depends(security.HTTPBasic()),
):
    if credentials.username == "bob":
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="You're not Bob",
        headers={"WWW-Authenticate": "Basic"},
    )


@pytest_asyncio.fixture()
async def route_dependencies_app():
    """Fixture to get the FastAPI app with custom route dependencies."""
    if os.getenv("BACKEND", "elasticsearch").lower() == "opensearch":
        from stac_fastapi.opensearch.app import instantiate_api as instantiate_api_local
    else:
        from stac_fastapi.elasticsearch.app import (
            instantiate_api as instantiate_api_local,
        )

    api = instantiate_api_local()
    app = api.app

    # Note: Route dependencies are already configured in the app via instantiate_api
    # We cannot add middleware after the app has started, so return as-is
    return app


@pytest_asyncio.fixture()
async def route_dependencies_client(route_dependencies_app):
    await create_index_templates()
    await create_collection_index()

    async with AsyncClient(
        transport=ASGITransport(app=route_dependencies_app),
        base_url="http://test-server",
    ) as c:
        yield c


def build_test_app(settings=None):
    """Build a test app with configurable transaction extensions."""
    if os.getenv("BACKEND", "elasticsearch").lower() == "opensearch":
        from stac_fastapi.opensearch.app import instantiate_api as instantiate_api_local
        from stac_fastapi.opensearch.config import OpensearchSettings

        settings_class = OpensearchSettings
    else:
        from stac_fastapi.elasticsearch.app import (
            instantiate_api as instantiate_api_local,
        )
        from stac_fastapi.elasticsearch.config import ElasticsearchSettings

        settings_class = ElasticsearchSettings

    if settings is None:
        settings = settings_class()

    api = instantiate_api_local(settings=settings)
    return api.app


def build_test_app_with_catalogs():
    """Build a test app with catalogs extension enabled."""
    if os.getenv("BACKEND", "elasticsearch").lower() == "opensearch":
        from stac_fastapi.opensearch.app import instantiate_api as instantiate_api_local
    else:
        from stac_fastapi.elasticsearch.app import (
            instantiate_api as instantiate_api_local,
        )

    api = instantiate_api_local()
    return api.app


@pytest_asyncio.fixture()
async def catalogs_app():
    """Fixture to get the FastAPI app with catalogs extension enabled."""
    return build_test_app_with_catalogs()


@pytest_asyncio.fixture()
async def catalogs_app_client(catalogs_app):
    """Fixture to get an async client for the app with catalogs extension enabled."""
    await create_index_templates()
    await create_collection_index()

    async with AsyncClient(
        transport=ASGITransport(app=catalogs_app), base_url="http://test-server"
    ) as c:
        yield c


def get_flattened_routes(router_obj, prefix=""):
    """
    Recursively extracts all flattened routes from a FastAPI app,
    navigating through Mounts, APIRouters, and FastAPI >= 0.137 _IncludedRouters.
    """
    api_routes = set()
    routes = getattr(router_obj, "routes", [])

    for route in routes:
        # 1. Standard Endpoints (APIRoute)
        if hasattr(route, "methods") and route.methods:
            for m in route.methods:
                if m == "HEAD":
                    continue
                r_path = getattr(route, "path", "")
                full_path = f"{prefix}{r_path}".replace("//", "/")
                api_routes.add(f"{m} {full_path}")

        # 2. Recurse into Mounts (Starlette)
        if hasattr(route, "app") and hasattr(route.app, "routes"):
            r_path = getattr(route, "path", getattr(route, "prefix", ""))
            next_prefix = f"{prefix}{r_path}"
            api_routes.update(get_flattened_routes(route.app, next_prefix))

        # 3. Recurse into FastAPI >= 0.137 _IncludedRouter wrappers
        if hasattr(route, "original_router"):
            r_prefix = getattr(route, "prefix", "")
            if not r_prefix and hasattr(route, "include_context"):
                r_prefix = getattr(route.include_context, "prefix", "")
            next_prefix = f"{prefix}{r_prefix}"
            api_routes.update(get_flattened_routes(route.original_router, next_prefix))

        # 4. Recurse into classic FastAPI/Starlette Routers (< 0.137)
        elif hasattr(route, "routes") and route is not router_obj:
            r_path = getattr(route, "path", getattr(route, "prefix", ""))
            next_prefix = f"{prefix}{r_path}"
            api_routes.update(get_flattened_routes(route, next_prefix))

    return api_routes


@pytest_asyncio.fixture()
async def mock_datetime_env(txn_client, monkeypatch):
    """Set USE_DATETIME environment variable to False for testing."""
    monkeypatch.setenv("USE_DATETIME", "false")
    if hasattr(txn_client.database.async_index_selector, "cache_manager"):
        await txn_client.database.async_index_selector.cache_manager.clear_cache()
    yield
    monkeypatch.setenv("USE_DATETIME", "true")
