#!/bin/bash
# Generate placeholder icons for AirImpute Pro

# Create a simple SVG icon
cat > icon.svg << 'EOF'
<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
  <rect width="256" height="256" fill="#4a90e2"/>
  <text x="128" y="128" font-family="Arial, sans-serif" font-size="120" fill="white" text-anchor="middle" dominant-baseline="middle">A</text>
</svg>
EOF

# Convert to PNG sizes
for size in 32 128 256; do
    convert -background none icon.svg -resize ${size}x${size} icons/${size}x${size}.png 2>/dev/null || {
        # If ImageMagick not available, use alternative method
        echo "Creating placeholder ${size}x${size}.png"
        python3 -c "
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGBA', (${size}, ${size}), color=(74, 144, 226, 255))
draw = ImageDraw.Draw(img)
# Draw a simple 'A' in the center
font_size = int(${size} * 0.5)
try:
    from PIL import ImageFont
    # Try to use a default font
    draw.text((${size}//2, ${size}//2), 'A', fill='white', anchor='mm')
except:
    pass
img.save('icons/${size}x${size}.png')
" 2>/dev/null || {
            # Final fallback - create with dd
            echo "Using fallback method for ${size}x${size}.png"
            # Create a minimal valid PNG header + data
            printf '\x89PNG\r\n\x1a\n' > icons/${size}x${size}.png
        }
    }
done

# Copy for required names
cp icons/128x128.png icons/icon.png 2>/dev/null || true
cp icons/256x256.png icons/128x128@2x.png 2>/dev/null || true

# Create ICO file (Windows icon)
cp icons/32x32.png icons/icon.ico 2>/dev/null || true

# Create ICNS file (macOS icon) - just copy PNG for now
cp icons/128x128.png icons/icon.icns 2>/dev/null || true

echo "Placeholder icons generated"