/**
 * Theme constants for scientific visualization
 */

export const SCIENTIFIC_COLORS = {
  // Diverging color schemes
  diverging: {
    RdBu: ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'],
    PRGn: ['#40004b', '#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837', '#00441b'],
    PiYG: ['#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419'],
  },
  
  // Sequential color schemes
  sequential: {
    Blues: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
    Greens: ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
    Reds: ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],
  },
  
  // Categorical colors
  categorical: {
    Set1: ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'],
    Set2: ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'],
    Set3: ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9'],
  },
  
  // Heatmap colors
  heatmap: {
    viridis: ['#440154', '#482677', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
    plasma: ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],
    inferno: ['#000004', '#1b0c41', '#4a0c4e', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#f7d03c', '#fcffa4'],
  },
} as const;

export const THEME_CONFIG = {
  light: {
    background: '#ffffff',
    foreground: '#000000',
    card: '#f8f9fa',
    border: '#e1e4e8',
    primary: '#0366d6',
    secondary: '#586069',
    muted: '#6a737d',
    accent: '#f6f8fa',
    success: '#28a745',
    warning: '#ffc107',
    error: '#dc3545',
    info: '#17a2b8',
  },
  dark: {
    background: '#0d1117',
    foreground: '#c9d1d9',
    card: '#161b22',
    border: '#30363d',
    primary: '#58a6ff',
    secondary: '#8b949e',
    muted: '#6e7681',
    accent: '#21262d',
    success: '#3fb950',
    warning: '#d29922',
    error: '#f85149',
    info: '#58a6ff',
  },
} as const;

// Define default typography and accessibility to ensure consistency
const DEFAULT_TYPOGRAPHY = {
  scientificNotation: false,
  significantFigures: 3,
  numberFormat: 'decimal' as const,
};

const DEFAULT_ACCESSIBILITY = {
  highContrast: false,
  colorBlindMode: 'none' as const,
  fontSize: 'medium' as const,
};

// Select default scientific palettes to be used in the themes
const BASE_SCIENTIFIC_VISUALIZATION_COLORS = {
  heatmapColors: [...SCIENTIFIC_COLORS.heatmap.viridis],
  divergingColors: [...SCIENTIFIC_COLORS.diverging.RdBu],
  categoricalColors: [...SCIENTIFIC_COLORS.categorical.Set1],
};

export const LIGHT_THEME = {
  mode: 'light' as const,
  colorScheme: {
    primary: THEME_CONFIG.light.primary,
    secondary: THEME_CONFIG.light.secondary,
    accent: THEME_CONFIG.light.accent,
    success: THEME_CONFIG.light.success,
    warning: THEME_CONFIG.light.warning,
    error: THEME_CONFIG.light.error,
    info: THEME_CONFIG.light.info,
    ...BASE_SCIENTIFIC_VISUALIZATION_COLORS,
  },
  typography: DEFAULT_TYPOGRAPHY,
  accessibility: DEFAULT_ACCESSIBILITY,
};

export const DARK_THEME = {
  mode: 'dark' as const,
  colorScheme: {
    primary: THEME_CONFIG.dark.primary,
    secondary: THEME_CONFIG.dark.secondary,
    accent: THEME_CONFIG.dark.accent,
    success: THEME_CONFIG.dark.success,
    warning: THEME_CONFIG.dark.warning,
    error: THEME_CONFIG.dark.error,
    info: THEME_CONFIG.dark.info,
    ...BASE_SCIENTIFIC_VISUALIZATION_COLORS,
  },
  typography: DEFAULT_TYPOGRAPHY,
  accessibility: DEFAULT_ACCESSIBILITY,
};

export const HIGH_CONTRAST_THEME = {
  mode: 'dark' as const,
  colorScheme: {
    primary: '#00ffff',
    secondary: '#ffff00',
    accent: '#333333',
    success: '#00ff00',
    warning: '#ffff00',
    error: '#ff0000',
    info: '#00ffff',
    ...BASE_SCIENTIFIC_VISUALIZATION_COLORS,
  },
  typography: DEFAULT_TYPOGRAPHY,
  accessibility: {
    ...DEFAULT_ACCESSIBILITY,
    highContrast: true,
  },
};