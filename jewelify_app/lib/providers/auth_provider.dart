import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class UserOut {
  final String id;
  final String? username;
  final String mobileNo;
  final String? createdAt;
  final String? accessToken;

  UserOut({
    required this.id,
    this.username,
    required this.mobileNo,
    this.createdAt,
    this.accessToken,
  });

  factory UserOut.fromJson(Map<String, dynamic> json) {
    final id = json['id'] as String?;
    final mobileNo = json['mobileNo'] as String?;
    if (id == null || mobileNo == null) {
      throw FormatException('Invalid JSON: id and mobileNo are required');
    }
    return UserOut(
      id: id,
      username: json['username'] as String?,
      mobileNo: mobileNo,
      createdAt: json['created_at'] as String?,
      accessToken: json['access_token'] as String?,
    );
  }
}

class AuthProvider with ChangeNotifier {
  String? _token;
  String? _userId;
  String? _username;
  String? _mobileNo;

  final _storage = const FlutterSecureStorage();

  String? get token => _token;
  String? get userId => _userId;
  String? get username => _username;
  String? get mobileNo => _mobileNo;
  bool get isAuthenticated => _token != null;

  AuthProvider() {
    loadToken();
  }

  Future<void> loadToken() async {
    _token = await _storage.read(key: 'auth_token');
    _userId = await _storage.read(key: 'user_id');
    _username = await _storage.read(key: 'username');
    _mobileNo = await _storage.read(key: 'mobileNo');
    notifyListeners();
  }

  Future<void> _saveToken({
    String? token,
    String? userId,
    String? username,
    String? mobileNo,
  }) async {
    if (token != null) await _storage.write(key: 'auth_token', value: token);
    if (userId != null) await _storage.write(key: 'user_id', value: userId);
    if (username != null) {
      await _storage.write(key: 'username', value: username);
    }
    if (mobileNo != null) {
      await _storage.write(key: 'mobileNo', value: mobileNo);
    }
  }

  Future<void> updateUserDetails({
    required String token,
    required String userId,
    required String username,
    required String mobileNo,
  }) async {
    _token = token;
    _userId = userId;
    _username = username;
    _mobileNo = mobileNo;
    await _saveToken(
      token: token,
      userId: userId,
      username: username,
      mobileNo: mobileNo,
    );
    notifyListeners();
  }

  Future<void> sendOtp(String mobileNo) async {
    try {
      final url = 'https://jewelify-server.onrender.com/auth/send-otp';
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'mobileNo': mobileNo}),
      );

      if (response.statusCode != 200) {
        final errorDetail =
            jsonDecode(response.body)['detail'] ?? 'Unknown error';
        throw Exception('Failed to send OTP: $errorDetail');
      }
    } catch (e) {
      rethrow;
    }
  }

  Future<void> verifyOtp(String mobileNo, String otp) async {
    try {
      final url = 'https://jewelify-server.onrender.com/auth/verify-otp';
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'mobileNo': mobileNo, 'otp': otp}),
      );

      if (response.statusCode != 200) {
        final errorDetail =
            jsonDecode(response.body)['detail'] ?? 'Unknown error';
        throw Exception('Failed to verify OTP: $errorDetail');
      }
    } catch (e) {
      rethrow;
    }
  }

  Future<void> login(String usernameOrMobile, String password) async {
    try {
      final url = 'https://jewelify-server.onrender.com/auth/login';
      final body = {'username': usernameOrMobile, 'password': password}.entries
          .map((e) => '${e.key}=${Uri.encodeQueryComponent(e.value)}')
          .join('&');

      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: body,
      );

      if (response.statusCode != 200) {
        final errorData = jsonDecode(response.body);
        throw Exception(
          'Failed to login: ${errorData['detail'] ?? response.body}',
        );
      }

      final data = jsonDecode(response.body);
      _token = data['access_token'];
      // Since the login endpoint doesn't return user details, fetch them
      final userData = await _fetchUserDetails(data['access_token']);
      _userId = userData['id'];
      _username = userData['username'];
      _mobileNo = userData['mobileNo'];
      await _saveToken(
        token: _token!,
        userId: _userId,
        username: _username,
        mobileNo: _mobileNo,
      );
      notifyListeners();
    } catch (e) {
      rethrow;
    }
  }

  Future<Map<String, dynamic>> _fetchUserDetails(String token) async {
    final url = 'https://jewelify-server.onrender.com/auth/me';
    final response = await http.get(
      Uri.parse(url),
      headers: {
        'Authorization': 'Bearer $token',
        'Content-Type': 'application/json',
      },
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to fetch user details');
    }

    return jsonDecode(response.body);
  }

  Future<void> logout() async {
    _token = null;
    _userId = null;
    _username = null;
    _mobileNo = null;
    await _storage.deleteAll();
    notifyListeners();
  }
}
