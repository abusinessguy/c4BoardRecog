import qrcode

# Function to generate QR code
def generate_qr(data, filename):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill='black', back_color='white')
    img.save(filename)

# Generate QR codes for each corner
generate_qr('top_left', 'top_left.png')
generate_qr('top_right', 'top_right.png')
generate_qr('bottom_left', 'bottom_left.png')
generate_qr('bottom_right', 'bottom_right.png')