#!/usr/bin/env node

/**
 * Download fonts locally to ensure offline functionality
 * This script downloads Inter and JetBrains Mono fonts for local use
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

const FONTS_DIR = path.join(__dirname, '..', 'src', 'assets', 'fonts');

// Font URLs (from Google Fonts CDN - only used during build)
const FONTS = [
  {
    name: 'Inter-Regular',
    url: 'https://fonts.gstatic.com/s/inter/v12/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hiA.woff2',
    format: 'woff2'
  },
  {
    name: 'Inter-Medium',
    url: 'https://fonts.gstatic.com/s/inter/v12/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuI6fAZ9hiA.woff2',
    format: 'woff2'
  },
  {
    name: 'Inter-SemiBold',
    url: 'https://fonts.gstatic.com/s/inter/v12/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuGKYAZ9hiA.woff2',
    format: 'woff2'
  },
  {
    name: 'Inter-Bold',
    url: 'https://fonts.gstatic.com/s/inter/v12/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuFuYAZ9hiA.woff2',
    format: 'woff2'
  },
  {
    name: 'JetBrainsMono-Regular',
    url: 'https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKxjPVmUsaaDhw.woff2',
    format: 'woff2'
  },
  {
    name: 'JetBrainsMono-SemiBold',
    url: 'https://fonts.gstatic.com/s/jetbrainsmono/v18/tDba2o-flEEny0FZhsfKu5WU4xD-IQ-PuZJJXxfpAO-Lf1OQk6OThxPA.woff2',
    format: 'woff2'
  }
];

// Create fonts directory if it doesn't exist
if (!fs.existsSync(FONTS_DIR)) {
  fs.mkdirSync(FONTS_DIR, { recursive: true });
}

// Check if fonts already exist
const existingFonts = fs.readdirSync(FONTS_DIR);
if (existingFonts.length >= FONTS.length) {
  console.log('Fonts already downloaded. Skipping...');
  process.exit(0);
}

console.log('Downloading fonts for offline use...');

// Download each font
const downloadFont = (font) => {
  return new Promise((resolve, reject) => {
    const filePath = path.join(FONTS_DIR, `${font.name}.${font.format}`);
    
    if (fs.existsSync(filePath)) {
      console.log(`✓ ${font.name} already exists`);
      resolve();
      return;
    }

    const file = fs.createWriteStream(filePath);
    
    https.get(font.url, (response) => {
      response.pipe(file);
      
      file.on('finish', () => {
        file.close();
        console.log(`✓ Downloaded ${font.name}`);
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(filePath, () => {}); // Delete the file on error
      reject(err);
    });
  });
};

// Download all fonts
Promise.all(FONTS.map(downloadFont))
  .then(() => {
    console.log('\n✓ All fonts downloaded successfully!');
    console.log('The app can now run completely offline.');
  })
  .catch((err) => {
    console.error('\n✗ Error downloading fonts:', err);
    console.error('Please check your internet connection and try again.');
    process.exit(1);
  });