/**
 * Time Series Chart Component
 * Implements IEEE standards for scientific data visualization
 * WCAG 2.1 Level AA compliant
 */

import React, { useMemo, useCallback, useState } from 'react';
import Plot from 'react-plotly.js';
import type { Data, Layout, Config } from 'plotly.js';
import type { TimeSeriesChartProps, TimeSeriesDataPoint } from '@/types/components/visualization';
import { getScientificAriaProps, announce } from '@/lib/accessibility';
import { cn } from '@/utils/cn';
import { useStore } from '@/store';

export const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  data,
  xAxisLabel = 'Time',
  yAxisLabel = 'Value',
  showConfidenceIntervals = true,
  showImputedPoints = true,
  dateFormat = '%Y-%m-%d %H:%M',
  valueFormat,
  zoomable = true,
  pannable = true,
  showLegend = true,
  legendPosition = 'top',
  colorScheme = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
  onPointClick,
  onRangeSelect,
  width = 800,
  height = 400,
  responsive = true,
  exportable = true,
  interactive = true,
  className,
  'aria-label': ariaLabel,
  'aria-describedby': ariaDescribedBy,
  uiMode = 'researcher',
  testId = 'time-series-chart',
}) => {
  const theme = useStore((state) => state.theme);
  const [selectedRange, setSelectedRange] = useState<[Date, Date] | null>(null);
  
  // Process data for Plotly
  const traces = useMemo(() => {
    const isMultiSeries = Array.isArray(data[0]);
    const series = isMultiSeries ? data as TimeSeriesDataPoint[][] : [data as TimeSeriesDataPoint[]];
    
    const plotTraces: any[] = [];
    
    series.forEach((seriesData, seriesIndex) => {
      const sortedData = [...seriesData].sort((a, b) => 
        a.timestamp.getTime() - b.timestamp.getTime()
      );
      
      // Main data trace
      const mainTrace = {
        x: sortedData.map(d => d.timestamp),
        y: sortedData.map(d => d.value),
        type: 'scatter',
        mode: 'lines+markers',
        name: `Series ${seriesIndex + 1}`,
        line: {
          color: colorScheme[seriesIndex % colorScheme.length],
          width: 2,
        },
        marker: {
          size: 6,
          color: sortedData.map(d => {
            if (showImputedPoints && d.isImputed) {
              return 'rgba(255, 0, 0, 0.6)';
            }
            return colorScheme[seriesIndex % colorScheme.length];
          }),
          symbol: sortedData.map(d => d.isImputed ? 'circle-open' : 'circle'),
        },
        hovertemplate: '%{x}<br>%{y:.4f}<extra>%{fullData.name}</extra>',
      };
      
      plotTraces.push(mainTrace);
      
      // Confidence intervals
      if (showConfidenceIntervals && sortedData.some(d => d.uncertainty)) {
        const ciData = sortedData.filter(d => d.uncertainty);
        
        const upperBound = {
          x: ciData.map(d => d.timestamp),
          y: ciData.map(d => d.uncertainty!.upper),
          type: 'scatter',
          mode: 'lines',
          name: `${mainTrace.name} (Upper CI)`,
          line: {
            color: colorScheme[seriesIndex % colorScheme.length],
            width: 0,
          },
          showlegend: false,
          hoverinfo: 'skip',
        };
        
        const lowerBound = {
          x: ciData.map(d => d.timestamp),
          y: ciData.map(d => d.uncertainty!.lower),
          type: 'scatter',
          mode: 'lines',
          name: `${mainTrace.name} (Lower CI)`,
          line: {
            color: colorScheme[seriesIndex % colorScheme.length],
            width: 0,
          },
          fill: 'tonexty',
          fillcolor: `rgba(${hexToRgb(colorScheme[seriesIndex % colorScheme.length])}, 0.2)`,
          showlegend: false,
          hoverinfo: 'skip',
        };
        
        plotTraces.push(lowerBound, upperBound);
      }
    });
    
    return plotTraces;
  }, [data, showConfidenceIntervals, showImputedPoints, colorScheme]);
  
  // Layout configuration
  const layout = useMemo(() => {
    const baseLayout: Partial<Layout> = {
      title: '',
      xaxis: {
        title: xAxisLabel,
        type: 'date',
        tickformat: dateFormat,
        showgrid: true,
        gridcolor: theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        zeroline: false,
      },
      yaxis: {
        title: yAxisLabel,
        showgrid: true,
        gridcolor: theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        zeroline: true,
        tickformat: valueFormat ? undefined : '.4f',
      },
      showlegend: showLegend,
      legend: {
        x: legendPosition === 'left' ? 0 : legendPosition === 'right' ? 1 : 0.5,
        y: legendPosition === 'top' ? 1.1 : legendPosition === 'bottom' ? -0.1 : 0.5,
        xanchor: legendPosition === 'left' ? 'left' : legendPosition === 'right' ? 'right' : 'center',
        yanchor: legendPosition === 'top' ? 'bottom' : legendPosition === 'bottom' ? 'top' : 'middle',
        orientation: legendPosition === 'left' || legendPosition === 'right' ? 'v' : 'h',
      },
      hovermode: 'closest',
      dragmode: zoomable ? 'zoom' : pannable ? 'pan' : false,
      paper_bgcolor: theme === 'dark' ? '#1f2937' : '#ffffff',
      plot_bgcolor: theme === 'dark' ? '#111827' : '#f9fafb',
      font: {
        color: theme === 'dark' ? '#e5e7eb' : '#1f2937',
      },
      margin: {
        l: 60,
        r: 20,
        t: 40,
        b: 60,
      },
    };
    
    // Add range selector for expert mode
    if (uiMode === 'expert' && baseLayout.xaxis) {
      baseLayout.xaxis.rangeselector = {
        buttons: [
          { count: 1, label: '1d', step: 'day', stepmode: 'backward' },
          { count: 7, label: '1w', step: 'day', stepmode: 'backward' },
          { count: 1, label: '1m', step: 'month', stepmode: 'backward' },
          { count: 3, label: '3m', step: 'month', stepmode: 'backward' },
          { count: 1, label: '1y', step: 'year', stepmode: 'backward' },
          { step: 'all', label: 'All' },
        ],
      };
    }
    
    return baseLayout;
  }, [
    theme,
    xAxisLabel,
    yAxisLabel,
    dateFormat,
    valueFormat,
    showLegend,
    legendPosition,
    zoomable,
    pannable,
    uiMode,
  ]);
  
  // Configuration
  const config = useMemo(() => ({
    displayModeBar: interactive && uiMode !== 'student',
    displaylogo: false,
    modeBarButtonsToRemove: uiMode === 'researcher' ? ['sendDataToCloud'] : [],
    modeBarButtonsToAdd: exportable ? [] : ['toImage'],
    responsive,
    toImageButtonOptions: {
      format: 'png',
      filename: 'time_series_chart',
      height: height,
      width: width,
      scale: 2,
    },
  }), [interactive, uiMode, exportable, responsive, height, width]);
  
  // Event handlers
  const handleClick = useCallback((event: any) => {
    if (onPointClick && event.points && event.points[0]) {
      const point = event.points[0];
      const dataPoint: TimeSeriesDataPoint = {
        timestamp: new Date(point.x),
        value: point.y,
      };
      onPointClick(dataPoint);
      announce(`Selected point at ${point.x} with value ${point.y}`);
    }
  }, [onPointClick]);
  
  const handleSelected = useCallback((event: any) => {
    if (onRangeSelect && event.range && event.range.x) {
      const start = new Date(event.range.x[0]);
      const end = new Date(event.range.x[1]);
      setSelectedRange([start, end]);
      onRangeSelect(start, end);
      announce(`Selected range from ${start.toLocaleString()} to ${end.toLocaleString()}`);
    }
  }, [onRangeSelect]);
  
  // Helper function to convert hex to RGB
  const hexToRgb = (hex: string): string => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? 
      `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}` : 
      '0, 0, 0';
  };
  
  const chartAriaProps = getScientificAriaProps('chart', {
    label: ariaLabel || `Time series chart showing ${yAxisLabel} over ${xAxisLabel}`,
    descriptionId: ariaDescribedBy,
  });
  
  return (
    <div 
      className={cn('time-series-chart', className)}
      data-testid={testId}
      {...chartAriaProps}
    >
      <Plot
        data={traces as Data[]}
        layout={layout as Layout}
        config={config as Config}
        style={{ width: '100%', height: '100%' }}
        onClickAnnotation={handleClick}
        onClick={handleClick}
        onSelected={handleSelected}
        onRelayout={(event) => {
          if (event['xaxis.range[0]'] && event['xaxis.range[1]']) {
            announce('Chart view updated');
          }
        }}
      />
      
      {/* Screen reader description */}
      <div className="sr-only" id={`${testId}-description`}>
        <p>
          This time series chart displays {traces.length / (showConfidenceIntervals ? 3 : 1)} data series.
          {showConfidenceIntervals && ' Confidence intervals are shown as shaded areas.'}
          {showImputedPoints && ' Imputed values are displayed with open circles.'}
        </p>
        {selectedRange && (
          <p>
            Selected range: {selectedRange[0].toLocaleString()} to {selectedRange[1].toLocaleString()}
          </p>
        )}
      </div>
    </div>
  );
};

// Export with display name for debugging
TimeSeriesChart.displayName = 'TimeSeriesChart';