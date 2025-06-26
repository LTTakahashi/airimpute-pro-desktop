#!/usr/bin/env node
// Node.js script to read Python version configuration

const fs = require('fs');
const path = require('path');

const configPath = path.join(__dirname, '..', 'python-version.json');

if (!fs.existsSync(configPath)) {
    console.error(`Python version configuration not found at: ${configPath}`);
    process.exit(1);
}

try {
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    const property = process.argv[2] || 'version';
    
    switch (property) {
        case 'version':
            console.log(config.version);
            break;
        case 'major':
            console.log(config.major);
            break;
        case 'minor':
            console.log(config.minor);
            break;
        case 'patch':
            console.log(config.patch);
            break;
        case 'dll_name':
            console.log(config.dll_name);
            break;
        case 'lib_name':
            console.log(config.lib_name);
            break;
        case 'choco_package':
            console.log(config.choco_package);
            break;
        case 'embeddable_url':
            console.log(config.embeddable_url);
            break;
        case 'executable':
            const platform = process.argv[3] || process.platform;
            const osMap = {
                'win32': 'windows',
                'darwin': 'macos',
                'linux': 'linux'
            };
            console.log(config.executable[osMap[platform] || 'linux']);
            break;
        default:
            console.log(config.version);
    }
} catch (error) {
    console.error(`Failed to read Python version configuration: ${error}`);
    process.exit(1);
}