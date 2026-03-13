
import qrcode
from pathlib import Path

def generate_quishing_qr(url: str, output_path: str):
    """
    Generates a QR code for a given URL and saves it to the specified path.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    
    # Ensure directory exists
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    img.save(output_path)
    return output_path

if __name__ == "__main__":
    # Test generation
    generate_quishing_qr("https://untrusted-bank-verification.net/verify", "red_agent/assets/quishing/test_qr.png")
    print("Test QR generated at red_agent/assets/quishing/test_qr.png")
