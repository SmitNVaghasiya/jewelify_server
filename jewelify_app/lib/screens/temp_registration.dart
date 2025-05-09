import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter/services.dart';
import '../providers/auth_provider.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  _RegisterScreenState createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _formKey = GlobalKey<FormState>();
  String _username = '';
  String _mobileNo = '';
  String _password = '';
  String _otp = '';
  bool _obscurePassword = true;
  bool _isLoading = false;
  bool _isOtpSent = false;

  final _usernameController = TextEditingController();
  final _mobileNoController = TextEditingController();
  final _passwordController = TextEditingController();
  final _otpController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _usernameController.addListener(_checkFieldsAndSendOtp);
    _mobileNoController.addListener(_checkFieldsAndSendOtp);
    _passwordController.addListener(_checkFieldsAndSendOtp);
  }

  @override
  void dispose() {
    _usernameController.dispose();
    _mobileNoController.dispose();
    _passwordController.dispose();
    _otpController.dispose();
    super.dispose();
  }

  void _checkFieldsAndSendOtp() {
    if (_usernameController.text.isNotEmpty &&
        _mobileNoController.text.length == 10 &&
        _passwordController.text.isNotEmpty &&
        !_isLoading &&
        !_isOtpSent) {
      if (_formKey.currentState!.validate()) {
        _formKey.currentState!.save();
        _sendOtp();
      }
    }
  }

  Future<void> _sendOtp() async {
    setState(() => _isLoading = true);
    final fullMobileNo = '+91$_mobileNo';

    try {
      final authProvider = Provider.of<AuthProvider>(context, listen: false);
      await authProvider.sendOtp(fullMobileNo);
      setState(() {
        _isOtpSent = true;
        _isLoading = false;
      });
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('OTP sent successfully!')));
    } catch (e) {
      setState(() {
        _isOtpSent = false; // Ensure _isOtpSent remains false on failure
        _isLoading = false;
      });
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Failed to send OTP: $e')));
    }
  }

  Future<void> _verifyOtpAndRegister() async {
    if (!_formKey.currentState!.validate()) return;
    _formKey.currentState!.save();
    setState(() => _isLoading = true);

    try {
      final authProvider = Provider.of<AuthProvider>(context, listen: false);
      final fullMobileNo = '+91$_mobileNo';
      await authProvider.verifyOtp(fullMobileNo, _otp);
      await _registerWithBackend();
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Registration failed: $e')));
      setState(() => _isLoading = false);
    }
  }

  Future<void> _registerWithBackend() async {
    final url = 'https://jewelify-server.onrender.com/auth/register';
    final response = await http.post(
      Uri.parse(url),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'username': _username,
        'mobileNo': '+91$_mobileNo',
        'password': _password,
      }),
    );

    if (response.statusCode == 200 || response.statusCode == 201) {
      final data = jsonDecode(response.body);
      final authProvider = Provider.of<AuthProvider>(context, listen: false);
      await authProvider.updateUserDetails(
        token: data['access_token'],
        userId: data['id'],
        username: _username,
        mobileNo: '+91$_mobileNo',
      );

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Registration successful!')));
      Navigator.pushNamedAndRemoveUntil(context, '/home', (route) => false);
      setState(() => _isLoading = false);
      return;
    }

    final errorDetail = jsonDecode(response.body)['detail'] ?? response.body;
    if (errorDetail == "Mobile number already exists") {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Mobile number already registered. Please login.'),
        ),
      );
      Navigator.pushReplacementNamed(context, '/login');
    } else if (errorDetail == "Username already exists") {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Username already taken. Please choose another.'),
        ),
      );
      setState(() {
        _isOtpSent = false; // Allow retry
      });
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Registration failed: $errorDetail')),
      );
    }
    setState(() => _isLoading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 50),
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.grey.shade200,
                  ),
                  child: const Icon(
                    Icons.arrow_back_ios_new,
                    size: 16,
                    color: Colors.black,
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Center(
                child: Image.asset(
                  'assets/images/login_illustration.png',
                  height: 180,
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                'Sign-up',
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.black,
                ),
              ),
              const SizedBox(height: 30),
              TextFormField(
                controller: _usernameController,
                decoration: const InputDecoration(
                  hintText: 'Username',
                  hintStyle: TextStyle(color: Color(0xFF757575)),
                ),
                style: const TextStyle(color: Color(0xFF757575)),
                validator:
                    (value) =>
                        value!.isEmpty ? 'Please enter a username' : null,
                onSaved: (value) => _username = value!,
              ),
              const SizedBox(height: 20),
              TextFormField(
                controller: _mobileNoController,
                decoration: const InputDecoration(
                  hintText: 'Mobile No. (e.g., 9876543210)',
                  hintStyle: TextStyle(color: Color(0xFF757575)),
                ),
                keyboardType: TextInputType.phone,
                maxLength: 10,
                style: const TextStyle(color: Color(0xFF757575)),
                validator: (value) {
                  if (value!.isEmpty) return 'Please enter your mobile number';
                  if (!RegExp(r'^\d{10}$').hasMatch(value)) {
                    return 'Enter a valid 10-digit mobile number';
                  }
                  return null;
                },
                onSaved: (value) => _mobileNo = value!,
              ),
              const SizedBox(height: 20),
              TextFormField(
                controller: _passwordController,
                decoration: InputDecoration(
                  hintText: 'Password',
                  hintStyle: const TextStyle(color: Color(0xFF757575)),
                  suffixIcon: IconButton(
                    icon: Icon(
                      _obscurePassword
                          ? Icons.visibility_off
                          : Icons.visibility,
                      color: Colors.grey,
                    ),
                    onPressed:
                        () => setState(
                          () => _obscurePassword = !_obscurePassword,
                        ),
                  ),
                ),
                obscureText: _obscurePassword,
                style: const TextStyle(color: Color(0xFF757575)),
                validator: (value) {
                  if (value!.isEmpty) return 'Please enter your password';
                  if (value.length < 6) {
                    return 'Password must be at least 6 characters';
                  }
                  return null;
                },
                onSaved: (value) => _password = value!,
              ),
              if (_isOtpSent) ...[
                const SizedBox(height: 20),
                TextFormField(
                  controller: _otpController,
                  decoration: const InputDecoration(
                    hintText: 'Enter OTP',
                    hintStyle: TextStyle(color: Color(0xFF757575)),
                  ),
                  keyboardType: TextInputType.number,
                  maxLength: 6,
                  inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                  style: const TextStyle(color: Color(0xFF757575)),
                  validator: (value) {
                    if (value!.isEmpty) return 'Please enter the OTP';
                    if (value.length != 6) return 'OTP must be 6 digits';
                    return null;
                  },
                  onSaved: (value) => _otp = value!,
                  onChanged: (value) {
                    if (value.length == 6) {
                      FocusScope.of(context).unfocus();
                    }
                  },
                ),
              ],
              const SizedBox(height: 30),
              if (_isOtpSent)
                SizedBox(
                  width: double.infinity,
                  height: 50,
                  child:
                      _isLoading
                          ? const Center(child: CircularProgressIndicator())
                          : ElevatedButton(
                            onPressed: _verifyOtpAndRegister,
                            child: const Text('Verify OTP & Register'),
                          ),
                ),
              if (_isLoading && !_isOtpSent)
                const Center(child: CircularProgressIndicator()),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text(
                    'Already have an account? ',
                    style: TextStyle(color: Color(0xFF757575)),
                  ),
                  GestureDetector(
                    onTap: () => Navigator.pushNamed(context, '/login'),
                    child: const Text(
                      'Login',
                      style: TextStyle(
                        color: Color(0xFF2D4356),
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
