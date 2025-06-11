/**
 * Theme Provider Component
 * Implements theme system with accessibility support
 * WCAG 2.1 Level AA compliant
 */

import React, { createContext, useContext, useEffect, useState } from 'react';
import { ScientificTheme } from '@/types/components';
import { LIGHT_THEME, DARK_THEME, HIGH_CONTRAST_THEME } from '@/lib/constants/themes';
import { prefersColorScheme, prefersHighContrast } from '@/lib/accessibility';

interface ThemeContextType {
  theme: ScientificTheme;
  setTheme: (theme: ScientificTheme) => void;
  toggleTheme: () => void;
  applyColorBlindMode: (mode: ScientificTheme['accessibility']['colorBlindMode']) => void;
  setFontSize: (size: ScientificTheme['accessibility']['fontSize']) => void;
  setNumberFormat: (format: ScientificTheme['typography']['numberFormat']) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: React.ReactNode;
  defaultTheme?: ScientificTheme;
  storageKey?: string;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({
  children,
  defaultTheme,
  storageKey = 'airimpute-theme',
}) => {
  const [theme, setThemeState] = useState<ScientificTheme>(() => {
    // Try to load from localStorage
    try {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (error) {
      console.warn('Failed to load theme from storage:', error);
    }
    
    // Use system preferences
    if (prefersHighContrast()) {
      return HIGH_CONTRAST_THEME;
    }
    
    const colorScheme = prefersColorScheme();
    if (colorScheme === 'dark') {
      return DARK_THEME;
    }
    
    return defaultTheme || LIGHT_THEME;
  });
  
  // Apply theme to document
  useEffect(() => {
    const root = document.documentElement;
    
    // Apply color mode
    root.classList.remove('light', 'dark');
    root.classList.add(theme.mode);
    
    // Apply font size
    const fontSizeMap = {
      small: '14px',
      medium: '16px',
      large: '18px',
    };
    root.style.fontSize = fontSizeMap[theme.accessibility.fontSize];
    
    // Apply high contrast
    if (theme.accessibility.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }
    
    // Apply color blind mode
    if (theme.accessibility.colorBlindMode && theme.accessibility.colorBlindMode !== 'none') {
      root.classList.add(`colorblind-${theme.accessibility.colorBlindMode}`);
    } else {
      root.classList.remove('colorblind-protanopia', 'colorblind-deuteranopia', 'colorblind-tritanopia');
    }
    
    // Apply CSS variables for colors
    const colors = theme.colorScheme;
    root.style.setProperty('--color-primary', colors.primary);
    root.style.setProperty('--color-secondary', colors.secondary);
    root.style.setProperty('--color-accent', colors.accent);
    root.style.setProperty('--color-success', colors.success);
    root.style.setProperty('--color-warning', colors.warning);
    root.style.setProperty('--color-error', colors.error);
    root.style.setProperty('--color-info', colors.info);
  }, [theme]);
  
  // Save theme to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(theme));
    } catch (error) {
      console.warn('Failed to save theme to storage:', error);
    }
  }, [theme, storageKey]);
  
  const setTheme = (newTheme: ScientificTheme) => {
    setThemeState(newTheme);
  };
  
  const toggleTheme = () => {
    setThemeState(prev => {
      // Determine the next base theme
      const newThemeBase = prev.mode === 'light' ? DARK_THEME : LIGHT_THEME;
      
      // Create the new theme state by layering previous user-specific
      // settings over the new base theme
      return {
        ...newThemeBase,
        accessibility: {
          ...newThemeBase.accessibility,
          // Preserve user's customizations
          fontSize: prev.accessibility.fontSize,
          colorBlindMode: prev.accessibility.colorBlindMode,
          // Preserve high-contrast state independently of light/dark toggle
          highContrast: prev.accessibility.highContrast,
        },
        typography: {
          ...newThemeBase.typography,
          // Preserve user's customizations
          numberFormat: prev.typography.numberFormat,
        },
      };
    });
  };
  
  const applyColorBlindMode = (mode: ScientificTheme['accessibility']['colorBlindMode']) => {
    setThemeState(prev => ({
      ...prev,
      accessibility: {
        ...prev.accessibility,
        colorBlindMode: mode,
      },
    }));
  };
  
  const setFontSize = (size: ScientificTheme['accessibility']['fontSize']) => {
    setThemeState(prev => ({
      ...prev,
      accessibility: {
        ...prev.accessibility,
        fontSize: size,
      },
    }));
  };
  
  const setNumberFormat = (format: ScientificTheme['typography']['numberFormat']) => {
    setThemeState(prev => ({
      ...prev,
      typography: {
        ...prev.typography,
        numberFormat: format,
      },
    }));
  };
  
  return (
    <ThemeContext.Provider
      value={{
        theme,
        setTheme,
        toggleTheme,
        applyColorBlindMode,
        setFontSize,
        setNumberFormat,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};

// Export with display name for debugging
ThemeProvider.displayName = 'ThemeProvider';