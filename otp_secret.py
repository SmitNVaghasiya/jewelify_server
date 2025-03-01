import pyotp
secret = pyotp.random_base32()
print(secret)  # Example output: "JBSWY3DPEHPK3PXP"

