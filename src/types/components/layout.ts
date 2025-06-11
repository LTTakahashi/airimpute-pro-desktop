/**
 * Layout component type definitions
 * Following IEEE HCI guidelines for scientific software interfaces
 */

import { LayoutProps, UIMode } from './index';

// Main layout container
export interface ScientificLayoutProps extends LayoutProps {
  mode: UIMode;
  sidebar?: boolean;
  header?: boolean;
  footer?: boolean;
  fullscreen?: boolean;
  onModeChange?: (mode: UIMode) => void;
}

// Workspace layout for data analysis
export interface WorkspaceLayoutProps extends LayoutProps {
  panels: {
    id: string;
    type: 'data' | 'visualization' | 'controls' | 'results' | 'console';
    title: string;
    content: React.ReactNode;
    size?: number; // percentage or pixels
    minSize?: number;
    maxSize?: number;
    collapsible?: boolean;
    closable?: boolean;
    floating?: boolean;
  }[];
  layout: 'horizontal' | 'vertical' | 'grid' | 'tabs';
  resizable?: boolean;
  onLayoutChange?: (layout: any) => void;
  saveLayout?: boolean;
}

// Card layout for grouped content
export interface ScientificCardProps extends LayoutProps {
  title?: string;
  subtitle?: string;
  description?: string;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
  collapsible?: boolean;
  collapsed?: boolean;
  onCollapse?: (collapsed: boolean) => void;
  variant?: 'default' | 'outlined' | 'elevated';
  status?: 'default' | 'success' | 'warning' | 'error' | 'info';
  loading?: boolean;
}

// Section layout with scientific context
export interface SectionProps extends LayoutProps {
  title: string;
  description?: string;
  level?: 1 | 2 | 3 | 4;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
  collapsible?: boolean;
  helpText?: string;
  citations?: string[];
}

// Grid layout for data/results display
export interface DataGridLayoutProps extends Omit<LayoutProps, 'gap'> {
  columns: number | { xs?: number; sm?: number; md?: number; lg?: number; xl?: number };
  rows?: number;
  gap?: number | { x?: number; y?: number } | 'none' | 'small' | 'medium' | 'large';
  gridGap?: number | { x?: number; y?: number };
  alignItems?: 'start' | 'center' | 'end' | 'stretch';
  justifyItems?: 'start' | 'center' | 'end' | 'stretch';
  areas?: string[][];
  responsive?: boolean;
}

// Tab layout for multi-view content
export interface TabLayoutProps extends LayoutProps {
  tabs: {
    id: string;
    label: string;
    icon?: React.ReactNode;
    content: React.ReactNode;
    disabled?: boolean;
    badge?: string | number;
    closable?: boolean;
  }[];
  activeTab?: string;
  onChange?: (tabId: string) => void;
  orientation?: 'horizontal' | 'vertical';
  variant?: 'default' | 'pills' | 'underline';
  scrollable?: boolean;
}

// Split pane layout
export interface SplitPaneProps extends LayoutProps {
  orientation: 'horizontal' | 'vertical';
  primaryPanel: React.ReactNode;
  secondaryPanel: React.ReactNode;
  primaryMinSize?: number;
  primaryMaxSize?: number;
  secondaryMinSize?: number;
  secondaryMaxSize?: number;
  defaultSize?: number;
  onSizeChange?: (size: number) => void;
  resizerStyle?: React.CSSProperties;
}

// Accordion layout for collapsible sections
export interface AccordionLayoutProps extends LayoutProps {
  sections: {
    id: string;
    title: string;
    content: React.ReactNode;
    icon?: React.ReactNode;
    disabled?: boolean;
    defaultExpanded?: boolean;
  }[];
  multiple?: boolean;
  expanded?: string | string[];
  onChange?: (expanded: string | string[]) => void;
  variant?: 'default' | 'outlined' | 'separated';
}

// Modal/Dialog layout
export interface ModalLayoutProps extends LayoutProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  size?: 'small' | 'medium' | 'large' | 'fullscreen';
  actions?: React.ReactNode;
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
  showCloseButton?: boolean;
  preventScroll?: boolean;
  centered?: boolean;
}

// Sidebar layout
export interface SidebarLayoutProps extends LayoutProps {
  position: 'left' | 'right';
  width?: number | string;
  collapsible?: boolean;
  collapsed?: boolean;
  onCollapse?: (collapsed: boolean) => void;
  overlay?: boolean;
  persistent?: boolean;
  elevation?: number;
}

// Header layout
export interface HeaderLayoutProps extends LayoutProps {
  title?: string;
  subtitle?: string;
  logo?: React.ReactNode;
  navigation?: React.ReactNode;
  actions?: React.ReactNode;
  sticky?: boolean;
  transparent?: boolean;
  elevation?: number;
}

// Footer layout
export interface FooterLayoutProps extends LayoutProps {
  copyright?: string;
  links?: { label: string; href: string }[];
  version?: string;
  status?: React.ReactNode;
  sticky?: boolean;
}