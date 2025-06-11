/**
 * TimeSeriesChart Component Tests
 * Ensures scientific accuracy and accessibility compliance
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TimeSeriesChart } from '../scientific/TimeSeriesChart';
import { TimeSeriesDataPoint } from '@/types/components';

// Mock Plotly to avoid rendering issues in tests
vi.mock('react-plotly.js', () => ({
  default: ({ data, layout, config, onClick, onSelected }: any) => (
    <div
      data-testid="mock-plotly"
      data-traces={JSON.stringify(data)}
      data-layout={JSON.stringify(layout)}
      data-config={JSON.stringify(config)}
      onClick={() => onClick?.({ points: [{ x: '2024-01-01', y: 50 }] })}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          onSelected?.({ range: { x: ['2024-01-01', '2024-01-02'] } });
        }
      }}
      role="img"
      aria-label={layout?.title || 'Chart'}
    >
      Mock Plotly Chart
    </div>
  ),
}));

describe('TimeSeriesChart', () => {
  const mockData: TimeSeriesDataPoint[] = [
    {
      timestamp: new Date('2024-01-01T00:00:00'),
      value: 42.5,
      confidence: 0.95,
      isImputed: false,
    },
    {
      timestamp: new Date('2024-01-01T01:00:00'),
      value: 45.3,
      confidence: 0.85,
      isImputed: true,
      imputationMethod: 'RAH',
      uncertainty: {
        lower: 40.1,
        upper: 50.5,
      },
    },
    {
      timestamp: new Date('2024-01-01T02:00:00'),
      value: 48.7,
      confidence: 0.95,
      isImputed: false,
    },
  ];
  
  it('renders with required props', () => {
    render(<TimeSeriesChart data={mockData} />);
    expect(screen.getByTestId('time-series-chart')).toBeInTheDocument();
    expect(screen.getByTestId('mock-plotly')).toBeInTheDocument();
  });
  
  it('applies correct ARIA attributes', () => {
    render(
      <TimeSeriesChart
        data={mockData}
        aria-label="Custom chart label"
        aria-describedby="chart-description"
      />
    );
    
    const chart = screen.getByTestId('time-series-chart');
    expect(chart).toHaveAttribute('role', 'img');
    expect(chart).toHaveAttribute('aria-label', 'Custom chart label');
    expect(chart).toHaveAttribute('aria-describedby', 'chart-description');
  });
  
  it('displays confidence intervals when enabled', () => {
    render(<TimeSeriesChart data={mockData} showConfidenceIntervals={true} />);
    
    const plotly = screen.getByTestId('mock-plotly');
    const traces = JSON.parse(plotly.getAttribute('data-traces') || '[]');
    
    // Should have main trace + upper and lower bounds for confidence intervals
    expect(traces.length).toBeGreaterThan(1);
  });
  
  it('highlights imputed points when enabled', () => {
    render(<TimeSeriesChart data={mockData} showImputedPoints={true} />);
    
    const plotly = screen.getByTestId('mock-plotly');
    const traces = JSON.parse(plotly.getAttribute('data-traces') || '[]');
    
    // Check that imputed points have different markers
    const mainTrace = traces[0];
    expect(mainTrace.marker.symbol).toContain('circle-open');
  });
  
  it('handles point click events', async () => {
    const handleClick = vi.fn();
    render(
      <TimeSeriesChart
        data={mockData}
        onPointClick={handleClick}
      />
    );
    
    const plotly = screen.getByTestId('mock-plotly');
    fireEvent.click(plotly);
    
    await waitFor(() => {
      expect(handleClick).toHaveBeenCalledWith(
        expect.objectContaining({
          timestamp: expect.any(Date),
          value: expect.any(Number),
        })
      );
    });
  });
  
  it('handles range selection', async () => {
    const handleRangeSelect = vi.fn();
    render(
      <TimeSeriesChart
        data={mockData}
        onRangeSelect={handleRangeSelect}
      />
    );
    
    const plotly = screen.getByTestId('mock-plotly');
    fireEvent.keyDown(plotly, { key: 'Enter' });
    
    await waitFor(() => {
      expect(handleRangeSelect).toHaveBeenCalledWith(
        expect.any(Date),
        expect.any(Date)
      );
    });
  });
  
  it('adapts interface based on UI mode', () => {
    const { rerender } = render(
      <TimeSeriesChart data={mockData} uiMode="student" />
    );
    
    let config = JSON.parse(
      screen.getByTestId('mock-plotly').getAttribute('data-config') || '{}'
    );
    expect(config.displayModeBar).toBe(false);
    
    rerender(<TimeSeriesChart data={mockData} uiMode="expert" />);
    
    const layout = JSON.parse(
      screen.getByTestId('mock-plotly').getAttribute('data-layout') || '{}'
    );
    expect(layout.xaxis.rangeselector).toBeDefined();
  });
  
  it('provides screen reader descriptions', () => {
    render(
      <TimeSeriesChart
        data={mockData}
        showConfidenceIntervals={true}
        showImputedPoints={true}
      />
    );
    
    const description = screen.getByText(/This time series chart displays/);
    expect(description).toBeInTheDocument();
    expect(description).toHaveClass('sr-only');
    expect(description.textContent).toContain('Confidence intervals are shown');
    expect(description.textContent).toContain('Imputed values are displayed');
  });
  
  it('supports multi-series data', () => {
    const multiSeriesData: TimeSeriesDataPoint[][] = [
      mockData,
      mockData.map(d => ({ ...d, value: d.value * 1.2 })),
    ];
    
    render(<TimeSeriesChart data={multiSeriesData} />);
    
    const plotly = screen.getByTestId('mock-plotly');
    const traces = JSON.parse(plotly.getAttribute('data-traces') || '[]');
    
    // Should have at least 2 main traces
    expect(traces.filter((t: any) => t.mode === 'lines+markers').length).toBe(2);
  });
  
  it('formats values according to specification', () => {
    const valueFormatter = (value: number) => `${value.toFixed(1)} μg/m³`;
    
    render(
      <TimeSeriesChart
        data={mockData}
        valueFormat={valueFormatter}
        yAxisLabel="Concentration"
      />
    );
    
    const layout = JSON.parse(
      screen.getByTestId('mock-plotly').getAttribute('data-layout') || '{}'
    );
    
    expect(layout.yaxis.title).toBe('Concentration');
    expect(layout.yaxis.tickformat).toBeUndefined(); // Custom formatter overrides
  });
  
  it('respects theme settings', () => {
    // Mock store for theme
    vi.mock('@/store', () => ({
      useStore: () => 'dark',
    }));
    
    render(<TimeSeriesChart data={mockData} />);
    
    const layout = JSON.parse(
      screen.getByTestId('mock-plotly').getAttribute('data-layout') || '{}'
    );
    
    expect(layout.paper_bgcolor).toBe('#1f2937');
    expect(layout.plot_bgcolor).toBe('#111827');
  });
});

describe('TimeSeriesChart Accessibility', () => {
  const mockData: TimeSeriesDataPoint[] = [
    {
      timestamp: new Date('2024-01-01'),
      value: 42.5,
    },
  ];
  
  it('announces data point selection', async () => {
    const announcements: string[] = [];
    
    // Mock announce function
    vi.mock('@/lib/accessibility', async () => {
      const actual = await vi.importActual('@/lib/accessibility');
      return {
        ...actual,
        announce: (message: string) => announcements.push(message),
      };
    });
    
    render(
      <TimeSeriesChart
        data={mockData}
        onPointClick={() => {}}
      />
    );
    
    const plotly = screen.getByTestId('mock-plotly');
    fireEvent.click(plotly);
    
    await waitFor(() => {
      expect(announcements).toContain(
        expect.stringMatching(/Selected point at/)
      );
    });
  });
  
  it('supports keyboard navigation', async () => {
    const user = userEvent.setup();
    render(<TimeSeriesChart data={mockData} />);
    
    const plotly = screen.getByTestId('mock-plotly');
    await user.tab();
    
    expect(plotly).toHaveFocus();
  });
  
  it('maintains color contrast ratios', () => {
    render(<TimeSeriesChart data={mockData} />);
    
    const layout = JSON.parse(
      screen.getByTestId('mock-plotly').getAttribute('data-layout') || '{}'
    );
    
    // Verify grid colors have sufficient contrast
    expect(layout.xaxis.gridcolor).toMatch(/rgba\(.*0\.1\)/);
    expect(layout.yaxis.gridcolor).toMatch(/rgba\(.*0\.1\)/);
  });
});