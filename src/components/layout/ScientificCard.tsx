/**
 * Scientific Card Component
 * Implements IEEE HCI guidelines for grouped content
 * WCAG 2.1 Level AA compliant
 */

import React, { useState, useCallback } from 'react';
import type { ScientificCardProps } from '@/types/components/layout';
import { cn } from '@/utils/cn';
import { ChevronDown, ChevronUp, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { announce } from '@/lib/accessibility';

export const ScientificCard: React.FC<ScientificCardProps> = ({
  title,
  subtitle,
  description,
  icon,
  actions,
  children,
  collapsible = false,
  collapsed: controlledCollapsed,
  onCollapse,
  variant = 'default',
  status = 'default',
  loading = false,
  padding = 'medium',
  className,
  'aria-label': ariaLabel,
  'aria-describedby': ariaDescribedBy,
  testId = 'scientific-card',
}) => {
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const isControlled = controlledCollapsed !== undefined;
  const collapsed = isControlled ? controlledCollapsed : internalCollapsed;
  
  const handleToggleCollapse = useCallback(() => {
    if (isControlled) {
      onCollapse?.(!collapsed);
    } else {
      setInternalCollapsed(!collapsed);
    }
    announce(collapsed ? 'Card expanded' : 'Card collapsed');
  }, [collapsed, isControlled, onCollapse]);
  
  // Variant styles
  const variantStyles = {
    default: 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700',
    outlined: 'bg-transparent border-gray-300 dark:border-gray-600',
    elevated: 'bg-white dark:bg-gray-800 shadow-md border-transparent',
  };
  
  // Status styles
  const statusStyles = {
    default: '',
    success: 'border-green-500 dark:border-green-400',
    warning: 'border-yellow-500 dark:border-yellow-400',
    error: 'border-red-500 dark:border-red-400',
    info: 'border-blue-500 dark:border-blue-400',
  };
  
  // Status indicators
  const statusIndicators = {
    success: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500',
    info: 'bg-blue-500',
  };
  
  // Padding styles
  const paddingStyles = {
    none: 'p-0',
    small: 'p-3',
    medium: 'p-6',
    large: 'p-8',
  };
  
  return (
    <div
      className={cn(
        'scientific-card rounded-lg border transition-colors relative',
        variantStyles[variant],
        statusStyles[status],
        className
      )}
      data-testid={testId}
      aria-label={ariaLabel}
      aria-describedby={ariaDescribedBy}
    >
      {/* Status indicator bar */}
      {status !== 'default' && (
        <div 
          className={cn(
            'absolute top-0 left-0 right-0 h-1 rounded-t-lg',
            statusIndicators[status]
          )}
          aria-hidden="true"
        />
      )}
      
      {/* Header */}
      {(title || subtitle || icon || actions || collapsible) && (
        <div 
          className={cn(
            'flex items-start justify-between border-b border-gray-200 dark:border-gray-700',
            paddingStyles[padding],
            'pb-4'
          )}
        >
          <div className="flex items-start gap-3 flex-1">
            {icon && (
              <div className="flex-shrink-0 mt-0.5" aria-hidden="true">
                {icon}
              </div>
            )}
            
            <div className="flex-1">
              {title && (
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {title}
                </h3>
              )}
              {subtitle && (
                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  {subtitle}
                </p>
              )}
              {description && (
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                  {description}
                </p>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-2 ml-4">
            {actions && <div className="flex items-center gap-2">{actions}</div>}
            
            {collapsible && (
              <button
                onClick={handleToggleCollapse}
                className={cn(
                  'p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
                  'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
                )}
                aria-expanded={!collapsed}
                aria-controls={`${testId}-content`}
                aria-label={collapsed ? 'Expand card' : 'Collapse card'}
              >
                {collapsed ? (
                  <ChevronDown className="w-5 h-5" />
                ) : (
                  <ChevronUp className="w-5 h-5" />
                )}
              </button>
            )}
          </div>
        </div>
      )}
      
      {/* Content */}
      <AnimatePresence initial={false}>
        {!collapsed && (
          <motion.div
            id={`${testId}-content`}
            initial={collapsible ? { height: 0, opacity: 0 } : undefined}
            animate={{ height: 'auto', opacity: 1 }}
            exit={collapsible ? { height: 0, opacity: 0 } : undefined}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className={cn(paddingStyles[padding], title && 'pt-4')}>
              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
                  <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">
                    Loading...
                  </span>
                </div>
              ) : (
                children
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Loading overlay */}
      {loading && !collapsed && (
        <div 
          className="absolute inset-0 bg-white/50 dark:bg-gray-800/50 rounded-lg"
          aria-hidden="true"
        />
      )}
    </div>
  );
};

// Export with display name for debugging
ScientificCard.displayName = 'ScientificCard';