import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.requests import Request

from open_webui.utils.audit import AuditContext, AuditLevel, AuditLoggingMiddleware
from open_webui.utils.logger import file_format


async def _noop_app(scope, receive, send):
    return


def _build_request(
    headers: list[tuple[bytes, bytes]] | None = None,
    *,
    method: str = 'POST',
    path: str = '/api/v1/test',
) -> Request:
    request_headers = headers or [(b'user-agent', b'pytest')]
    scope = {
        'type': 'http',
        'method': method,
        'path': path,
        'headers': request_headers,
        'query_string': b'',
        'client': ('127.0.0.1', 0),
        'server': ('testserver', 80),
        'scheme': 'http',
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_log_audit_entry_logs_exception_with_request_context() -> None:
    middleware = AuditLoggingMiddleware(_noop_app, audit_level=AuditLevel.REQUEST)
    middleware._get_authenticated_user = AsyncMock(return_value=None)
    middleware.audit_logger.write = Mock(side_effect=RuntimeError('write failed'))

    request = _build_request()
    context = AuditContext()

    with patch('open_webui.utils.audit.logger.exception') as mock_exception:
        await middleware._log_audit_entry(request, context)

    mock_exception.assert_called_once_with(
        'Failed to log audit entry for {method} {path}',
        method='POST',
        path='/api/v1/test',
    )


@pytest.mark.asyncio
async def test_log_audit_entry_includes_request_identifiers() -> None:
    middleware = AuditLoggingMiddleware(_noop_app, audit_level=AuditLevel.REQUEST_RESPONSE)
    middleware._get_authenticated_user = AsyncMock(return_value=None)
    middleware.audit_logger.write = Mock()

    request = _build_request(
        headers=[
            (b'user-agent', b'pytest'),
            (b'x-request-id', b'req-123'),
            (b'x-correlation-id', b'corr-abc'),
        ]
    )
    context = AuditContext()
    context.metadata['response_status_code'] = 201

    await middleware._log_audit_entry(request, context)

    written_entry = middleware.audit_logger.write.call_args.args[0]
    assert written_entry.request_id == 'req-123'
    assert written_entry.correlation_id == 'corr-abc'
    assert written_entry.response_status_code == 201
    assert written_entry.verb == 'POST'


@pytest.mark.asyncio
async def test_log_audit_entry_redacts_password() -> None:
    middleware = AuditLoggingMiddleware(_noop_app, audit_level=AuditLevel.REQUEST)
    middleware._get_authenticated_user = AsyncMock(return_value=None)
    middleware.audit_logger.write = Mock()

    request = _build_request()
    context = AuditContext()
    context.add_request_chunk(b'{"password": "super-secret", "safe": "value"}')

    await middleware._log_audit_entry(request, context)

    written_entry = middleware.audit_logger.write.call_args.args[0]
    assert 'super-secret' not in written_entry.request_object
    assert '"password": "********"' in written_entry.request_object
    assert '"safe": "value"' in written_entry.request_object


def test_log_audit_entry_uses_empty_identifiers_when_missing_headers() -> None:
    middleware = AuditLoggingMiddleware(_noop_app, audit_level=AuditLevel.REQUEST)

    assert middleware.REQUEST_ID_HEADER == 'x-request-id'
    assert middleware.CORRELATION_ID_HEADER == 'x-correlation-id'


def test_file_format_includes_request_and_correlation_id() -> None:
    record = {
        'time': datetime(2026, 4, 5, 12, 0, 0),
        'extra': {
            'id': 'audit-1',
            'user': {'id': 'u-1'},
            'audit_level': 'REQUEST',
            'verb': 'POST',
            'request_uri': 'http://testserver/api/v1/test',
            'request_id': 'req-123',
            'correlation_id': 'corr-abc',
            'response_status_code': 200,
            'source_ip': '127.0.0.1',
            'user_agent': 'pytest',
            'request_object': '{"a": 1}',
            'response_object': '{"ok": true}',
            'extra': {'env': 'test'},
        },
    }

    formatted = file_format(record)
    payload = json.loads(record['extra']['file_extra'])

    assert formatted == '{extra[file_extra]}\n'
    assert payload['request_id'] == 'req-123'
    assert payload['correlation_id'] == 'corr-abc'
    assert payload['request_uri'] == 'http://testserver/api/v1/test'
    assert payload['response_status_code'] == 200


def test_should_skip_for_non_audited_method() -> None:
    middleware = AuditLoggingMiddleware(_noop_app, audit_level=AuditLevel.REQUEST)
    request = _build_request(
        headers=[(b'user-agent', b'pytest'), (b'authorization', b'Bearer t')],
        method='GET',
    )

    with patch('open_webui.utils.audit.AUDIT_LOG_LEVEL', 'REQUEST'):
        assert middleware._should_skip_auditing(request) is True


def test_should_not_skip_auth_endpoint_without_auth_headers() -> None:
    middleware = AuditLoggingMiddleware(_noop_app, audit_level=AuditLevel.REQUEST)
    request = _build_request(path='/api/v1/auths/signin')

    with patch('open_webui.utils.audit.AUDIT_LOG_LEVEL', 'REQUEST'):
        assert middleware._should_skip_auditing(request) is False


def test_should_skip_regular_endpoint_without_auth_headers() -> None:
    middleware = AuditLoggingMiddleware(_noop_app, audit_level=AuditLevel.REQUEST)
    request = _build_request(path='/api/v1/chats')

    with patch('open_webui.utils.audit.AUDIT_LOG_LEVEL', 'REQUEST'):
        assert middleware._should_skip_auditing(request) is True


def test_included_paths_logs_only_matching_prefix() -> None:
    middleware = AuditLoggingMiddleware(
        _noop_app,
        audit_level=AuditLevel.REQUEST,
        included_paths=['chats', 'models'],
    )
    matching_request = _build_request(
        path='/api/v1/chats',
        headers=[(b'user-agent', b'pytest'), (b'authorization', b'Bearer token')],
    )
    non_matching_request = _build_request(
        path='/api/v1/files',
        headers=[(b'user-agent', b'pytest'), (b'authorization', b'Bearer token')],
    )

    with patch('open_webui.utils.audit.AUDIT_LOG_LEVEL', 'REQUEST'):
        assert middleware._should_skip_auditing(matching_request) is False
        assert middleware._should_skip_auditing(non_matching_request) is True


def test_excluded_paths_skips_matching_prefix() -> None:
    middleware = AuditLoggingMiddleware(
        _noop_app,
        audit_level=AuditLevel.REQUEST,
        excluded_paths=['models'],
    )
    excluded_request = _build_request(
        path='/api/v1/models',
        headers=[(b'user-agent', b'pytest'), (b'authorization', b'Bearer token')],
    )
    allowed_request = _build_request(
        path='/api/v1/chats',
        headers=[(b'user-agent', b'pytest'), (b'authorization', b'Bearer token')],
    )

    with patch('open_webui.utils.audit.AUDIT_LOG_LEVEL', 'REQUEST'):
        assert middleware._should_skip_auditing(excluded_request) is True
        assert middleware._should_skip_auditing(allowed_request) is False
