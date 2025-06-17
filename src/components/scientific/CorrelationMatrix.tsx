/**
 * Correlation Matrix Component
 * Implements IEEE standards for statistical visualization
 * WCAG 2.1 Level AA compliant
 */

import React, { useMemo, useCallback } from 'react';
import Plot from 'react-plotly.js';
import type { Layout, Config } from 'plotly.js';
import type { CorrelationMatrixProps } from '@/types/components/visualization';
import { getScientificAriaProps, announce } from '@/lib/accessibility';
import { cn } from '@/utils/cn';
import { useStore } from '@/store';
import { SCIENTIFIC_COLORS } from '@/lib/constants/themes';

export const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({
  data,
  colorScale = 'diverging',
  showValues = true,
  showSignificance = true,
  significanceThreshold = 0.05,
  clusterVariables = false,
  onCellClick,
  annotateSignificant = true,
  maskInsignificant = false,
  width = 600,
  height = 600,
  responsive = true,
  interactive = true,
  className,
  'aria-label': ariaLabel,
  'aria-describedby': ariaDescribedBy,
  uiMode = 'researcher',
  testId = 'correlation-matrix',
}) => {
  const theme = useStore((state) => state.theme);
  
  // Process data for clustering if enabled
  const processedData = useMemo(() => {
    if (!clusterVariables) {
      return data;
    }
    
    // Simple hierarchical clustering implementation
    // For production, use a proper clustering library
    const n = data.variables.length;
    const distances: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));
    
    // Calculate distance matrix (1 - |correlation|)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        distances[i][j] = 1 - Math.abs(data.values[i][j]);
      }
    }
    
    // Simple clustering (nearest neighbor)
    const order = hierarchicalCluster(distances);
    
    // Reorder variables and values
    const reorderedVariables = order.map(i => data.variables[i]);
    const reorderedValues = order.map(i => order.map(j => data.values[i][j]));
    const reorderedPValues = data.pValues ? 
      order.map(i => order.map(j => data.pValues![i][j])) : 
      undefined;
    
    return {
      variables: reorderedVariables,
      values: reorderedValues,
      pValues: reorderedPValues,
      significanceLevel: data.significanceLevel,
    };
  }, [data, clusterVariables]);
  
  // Prepare heatmap data
  const heatmapData = useMemo(() => {
    const { variables, values, pValues } = processedData;
    
    // Apply masking if enabled
    const maskedValues = values.map((row, i) => 
      row.map((val, j) => {
        if (maskInsignificant && pValues && pValues[i][j] > significanceThreshold) {
          return NaN;
        }
        return val;
      })
    );
    
    // Prepare text annotations
    const textAnnotations = showValues ? values.map((row, i) => 
      row.map((val, j) => {
        let text = val.toFixed(3);
        if (showSignificance && pValues && annotateSignificant) {
          if (pValues[i][j] < 0.001) text += '***';
          else if (pValues[i][j] < 0.01) text += '**';
          else if (pValues[i][j] < 0.05) text += '*';
        }
        return text;
      })
    ) : undefined;
    
    return {
      z: maskedValues,
      x: variables,
      y: variables,
      text: textAnnotations,
      colorscale: colorScale === 'diverging' ? 
        SCIENTIFIC_COLORS.diverging.RdBu.map((color, i) => [
          i / (SCIENTIFIC_COLORS.diverging.RdBu.length - 1),
          color
        ]) :
        SCIENTIFIC_COLORS.sequential.Blues.map((color, i) => [
          i / (SCIENTIFIC_COLORS.sequential.Blues.length - 1),
          color
        ]),
      type: 'heatmap' as const,
      hovertemplate: 'Row: %{y}<br>Column: %{x}<br>Correlation: %{z:.3f}<extra></extra>',
      zmin: colorScale === 'diverging' ? -1 : 0,
      zmax: 1,
      showscale: true,
      colorbar: {
        title: 'Correlation',
        titleside: 'right',
        tickmode: 'linear',
        tick0: colorScale === 'diverging' ? -1 : 0,
        dtick: 0.2,
      },
    };
  }, [processedData, colorScale, showValues, showSignificance, significanceThreshold, 
      annotateSignificant, maskInsignificant]);
  
  // Layout configuration
  const layout = useMemo(() => ({
    title: '',
    xaxis: {
      title: '',
      side: 'bottom' as const,
      tickangle: -45,
      tickfont: {
        size: uiMode === 'student' ? 12 : 10,
      },
    },
    yaxis: {
      title: '',
      autorange: 'reversed' as const,
      tickfont: {
        size: uiMode === 'student' ? 12 : 10,
      },
    },
    paper_bgcolor: theme === 'dark' ? '#1f2937' : '#ffffff',
    plot_bgcolor: theme === 'dark' ? '#111827' : '#f9fafb',
    font: {
      color: theme === 'dark' ? '#e5e7eb' : '#1f2937',
    },
    margin: {
      l: 100,
      r: 80,
      t: 20,
      b: 100,
    },
    annotations: showValues ? heatmapData.text?.flatMap((row, i) => 
      row.map((text, j) => ({
        x: heatmapData.x[j],
        y: heatmapData.y[i],
        text: text,
        showarrow: false,
        font: {
          size: 9,
          color: Math.abs(heatmapData.z[i][j]) > 0.5 ? 'white' : 
                 theme === 'dark' ? '#e5e7eb' : '#1f2937',
        },
      }))
    ) : [],
  }), [theme, uiMode, showValues, heatmapData]);
  
  // Configuration
  const config = useMemo(() => ({
    displayModeBar: interactive && uiMode !== 'student',
    displaylogo: false,
    modeBarButtonsToRemove: ['sendDataToCloud'],
    responsive,
    toImageButtonOptions: {
      format: 'png' as const,
      filename: 'correlation_matrix',
      height: height,
      width: width,
      scale: 2,
    },
  }), [interactive, uiMode, responsive, height, width]);
  
  // Event handlers
  const handleClick = useCallback((event: any) => {
    if (onCellClick && event.points && event.points[0]) {
      const point = event.points[0];
      const row = point.y;
      const col = point.x;
      const value = point.z;
      
      onCellClick(row, col, value);
      announce(`Selected correlation between ${row} and ${col}: ${value.toFixed(3)}`);
    }
  }, [onCellClick]);
  
  // Helper function for hierarchical clustering
  const hierarchicalCluster = (distances: number[][]): number[] => {
    const n = distances.length;
    const clusters: number[][] = Array(n).fill(null).map((_, i) => [i]);
    
    // Simple agglomerative clustering
    while (clusters.length > 1) {
      let minDist = Infinity;
      let mergeI = 0;
      let mergeJ = 1;
      
      // Find closest clusters
      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          const dist = getClusterDistance(distances, clusters[i], clusters[j]);
          if (dist < minDist) {
            minDist = dist;
            mergeI = i;
            mergeJ = j;
          }
        }
      }
      
      // Merge clusters
      clusters[mergeI] = [...clusters[mergeI], ...clusters[mergeJ]];
      clusters.splice(mergeJ, 1);
    }
    
    return clusters[0];
  };
  
  const getClusterDistance = (
    distances: number[][], 
    cluster1: number[], 
    cluster2: number[]
  ): number => {
    // Average linkage
    let sum = 0;
    let count = 0;
    
    for (const i of cluster1) {
      for (const j of cluster2) {
        sum += distances[i][j];
        count++;
      }
    }
    
    return sum / count;
  };
  
  const matrixAriaProps = getScientificAriaProps('chart', {
    label: ariaLabel || `Correlation matrix with ${data.variables.length} variables`,
    descriptionId: ariaDescribedBy,
  });
  
  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    const flatValues = processedData.values.flat();
    const validValues = flatValues.filter(v => !isNaN(v));
    
    return {
      count: validValues.length,
      mean: validValues.reduce((a, b) => a + b, 0) / validValues.length,
      min: Math.min(...validValues),
      max: Math.max(...validValues),
      significant: processedData.pValues ? 
        processedData.pValues.flat().filter(p => p < significanceThreshold).length : 0,
    };
  }, [processedData, significanceThreshold]);
  
  return (
    <div 
      className={cn('correlation-matrix', className)}
      data-testid={testId}
      {...matrixAriaProps}
    >
      <Plot
        // HACK: @types/plotly.js has incorrect type for heatmap.text. It expects
        // string | string[] but the library requires string[][] for cell text.
        // See: https://plotly.com/javascript/reference/heatmap/#heatmap-text
        data={[heatmapData as any]}
        layout={layout as Layout}
        config={config as Config}
        style={{ width: '100%', height: '100%' }}
        onClick={handleClick}
        onHover={(event) => {
          if (event.points && event.points[0]) {
            const point = event.points[0];
            announce(`Hovering over correlation between ${point.y} and ${point.x}`, 'polite');
          }
        }}
      />
      
      {/* Screen reader description */}
      <div className="sr-only" id={`${testId}-description`}>
        <h3>Correlation Matrix Summary</h3>
        <p>
          This correlation matrix shows relationships between {processedData.variables.length} variables.
        </p>
        <ul>
          <li>Number of correlations: {summaryStats.count}</li>
          <li>Mean correlation: {summaryStats.mean.toFixed(3)}</li>
          <li>Range: {summaryStats.min.toFixed(3)} to {summaryStats.max.toFixed(3)}</li>
          {showSignificance && (
            <li>
              Significant correlations (p &lt; {significanceThreshold}): {summaryStats.significant}
            </li>
          )}
        </ul>
        {clusterVariables && (
          <p>Variables have been reordered using hierarchical clustering.</p>
        )}
      </div>
      
      {/* Legend for significance markers */}
      {showSignificance && annotateSignificant && uiMode !== 'student' && (
        <div 
          className="mt-2 text-sm text-gray-600 dark:text-gray-400"
          aria-label="Significance legend"
        >
          <span className="mr-4">* p &lt; 0.05</span>
          <span className="mr-4">** p &lt; 0.01</span>
          <span>*** p &lt; 0.001</span>
        </div>
      )}
    </div>
  );
};

// Export with display name for debugging
CorrelationMatrix.displayName = 'CorrelationMatrix';