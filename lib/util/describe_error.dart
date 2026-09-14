import 'dart:async';
import 'dart:io';

import '../api/ibvap_client.dart';

/// Turns transport / HTTP failures into one short line an operator can act on,
/// instead of `IbvapApiException(null): Cannot reach … ClientException with
/// SocketException: …`.
String describeError(Object e) {
  if (e is IbvapApiException) {
    final code = e.statusCode;
    if (code == 401 || code == 403) {
      return 'Write access denied — set the API token in Settings › Server.';
    }
    if (code == 404) return 'Not supported by this server version (HTTP 404).';
    if (code != null && code >= 500) return 'Server error (HTTP $code): ${_clip(e.message)}';
    if (code != null) return _clip(e.message);
    return _transport(e.cause ?? e.message);
  }
  return _transport(e);
}

String _transport(Object e) {
  final s = e.toString();
  if (e is TimeoutException || s.contains('TimeoutException')) {
    return 'Server not responding (timed out).';
  }
  if (e is SocketException ||
      s.contains('SocketException') ||
      s.contains('Connection refused') ||
      s.contains('actively refused')) {
    return 'Server unreachable — is the backend running?';
  }
  if (s.contains('HandshakeException') || s.contains('CERTIFICATE')) {
    return 'TLS handshake failed — check the server address scheme.';
  }
  if (s.contains('FormatException')) return 'Malformed response from server.';
  return _clip(s);
}

String _clip(String s, [int max = 140]) =>
    s.length <= max ? s : '${s.substring(0, max - 1)}…';
