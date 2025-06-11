import React from 'react';
import {
  BarChart as RechartsBarChart,
  Bar,
  LineChart as RechartsLineChart,
  Line,
  ScatterChart as RechartsScatterChart,
  Scatter,
  RadarChart as RechartsRadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  Area,
  ComposedChart,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { cn } from '@/utils/cn';

interface ChartProps {
  data: any;
  title?: string;
  className?: string;
  height?: number;
}

interface BarChartProps extends ChartProps {
  yAxisLabel?: string;
  xAxisLabel?: string;
  stacked?: boolean;
}

export const BarChart: React.FC<BarChartProps> = ({
  data,
  title,
  yAxisLabel,
  xAxisLabel,
  stacked = false,
  className,
  height = 400,
}) => {
  const colors = ['#3b82f6', '#ec4899', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444'];

  return (
    <Card className={className}>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <RechartsBarChart data={data.labels.map((label: string, idx: number) => ({
            name: label,
            ...data.datasets.reduce((acc: any, dataset: any, dsIdx: number) => ({
              ...acc,
              [dataset.label]: dataset.data[idx]
            }), {})
          }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" label={{ value: xAxisLabel, position: 'insideBottom', offset: -5 }} />
            <YAxis label={{ value: yAxisLabel, angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            {data.datasets.map((dataset: any, idx: number) => (
              <Bar
                key={dataset.label}
                dataKey={dataset.label}
                fill={dataset.backgroundColor || colors[idx % colors.length]}
                stackId={stacked ? "stack" : undefined}
              />
            ))}
          </RechartsBarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

interface LineChartProps extends ChartProps {
  yAxisLabel?: string;
  xAxisLabel?: string;
  showArea?: boolean;
}

export const LineChart: React.FC<LineChartProps> = ({
  data,
  title,
  yAxisLabel,
  xAxisLabel,
  showArea = false,
  className,
  height = 400,
}) => {
  const colors = ['#3b82f6', '#ec4899', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444'];

  return (
    <Card className={className}>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <RechartsLineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="x" 
              label={{ value: xAxisLabel, position: 'insideBottom', offset: -5 }}
              domain={['dataMin', 'dataMax']}
            />
            <YAxis label={{ value: yAxisLabel, angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            {data.datasets?.map((dataset: any, idx: number) => (
              <Line
                key={dataset.label}
                type="monotone"
                dataKey="y"
                data={dataset.data}
                stroke={colors[idx % colors.length]}
                fill={colors[idx % colors.length]}
                fillOpacity={showArea ? 0.3 : 0}
                strokeWidth={2}
                dot={false}
                name={dataset.label}
              />
            ))}
          </RechartsLineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

interface ScatterChartProps extends ChartProps {
  xAxisLabel?: string;
  yAxisLabel?: string;
}

export const ScatterChart: React.FC<ScatterChartProps> = ({
  data,
  title,
  xAxisLabel,
  yAxisLabel,
  className,
  height = 400,
}) => {
  const colors = ['#3b82f6', '#ec4899', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444'];

  return (
    <Card className={className}>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <RechartsScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="x"
              type="number"
              label={{ value: xAxisLabel, position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              dataKey="y"
              type="number"
              label={{ value: yAxisLabel, angle: -90, position: 'insideLeft' }}
            />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend />
            {data.datasets.map((dataset: any, idx: number) => (
              <Scatter
                key={dataset.label}
                name={dataset.label}
                data={dataset.data}
                fill={colors[idx % colors.length]}
              />
            ))}
          </RechartsScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export const RadarChart: React.FC<ChartProps> = ({
  data,
  title,
  className,
  height = 400,
}) => {
  const colors = ['#3b82f6', '#ec4899', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444'];

  const radarData = data.labels.map((label: string, idx: number) => ({
    metric: label,
    ...data.datasets.reduce((acc: any, dataset: any) => ({
      ...acc,
      [dataset.label]: dataset.data[idx]
    }), {})
  }));

  return (
    <Card className={className}>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <RechartsRadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" />
            <PolarRadiusAxis angle={90} domain={[0, 1]} />
            {data.datasets.map((dataset: any, idx: number) => (
              <Radar
                key={dataset.label}
                name={dataset.label}
                dataKey={dataset.label}
                stroke={colors[idx % colors.length]}
                fill={colors[idx % colors.length]}
                fillOpacity={0.3}
              />
            ))}
            <Legend />
          </RechartsRadarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

interface HeatmapChartProps extends ChartProps {
  colorScale?: 'viridis' | 'plasma' | 'inferno' | 'magma' | 'cividis';
}

export const HeatmapChart: React.FC<HeatmapChartProps> = ({
  data,
  title,
  colorScale = 'viridis',
  className,
  height = 400,
}) => {
  // Color scales approximations
  const colorScales = {
    viridis: ['#440154', '#3e4989', '#26828e', '#35b779', '#fde725'],
    plasma: ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636'],
    inferno: ['#000004', '#420a68', '#932667', '#dd513a', '#fcffa4'],
    magma: ['#000004', '#3b0f70', '#8c2981', '#de4968', '#fcfdbf'],
    cividis: ['#00204d', '#00306f', '#165086', '#3e6e8e', '#7c9885'],
  };

  const colors = colorScales[colorScale];
  
  // Flatten heatmap data for rendering
  const heatmapData: any[] = [];
  data.methods.forEach((method: string, mIdx: number) => {
    data.datasets.forEach((dataset: string, dIdx: number) => {
      const value = data.values[mIdx][dIdx];
      if (value !== null) {
        heatmapData.push({
          method,
          dataset,
          value,
          x: dIdx,
          y: mIdx,
        });
      }
    });
  });

  const minValue = Math.min(...heatmapData.map(d => d.value));
  const maxValue = Math.max(...heatmapData.map(d => d.value));

  const getColor = (value: number) => {
    const normalized = (value - minValue) / (maxValue - minValue);
    const colorIndex = Math.floor(normalized * (colors.length - 1));
    return colors[colorIndex];
  };

  return (
    <Card className={className}>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <div className="relative" style={{ height }}>
          <svg width="100%" height="100%" viewBox="0 0 500 400">
            <g transform="translate(80, 40)">
              {/* Y-axis labels (methods) */}
              {data.methods.map((method: string, idx: number) => (
                <text
                  key={method}
                  x="-10"
                  y={idx * (300 / data.methods.length) + 15}
                  textAnchor="end"
                  fontSize="12"
                  fill="currentColor"
                >
                  {method}
                </text>
              ))}
              
              {/* X-axis labels (datasets) */}
              {data.datasets.map((dataset: string, idx: number) => (
                <text
                  key={dataset}
                  x={idx * (380 / data.datasets.length) + 40}
                  y="320"
                  textAnchor="middle"
                  fontSize="12"
                  fill="currentColor"
                  transform={`rotate(-45, ${idx * (380 / data.datasets.length) + 40}, 320)`}
                >
                  {dataset}
                </text>
              ))}
              
              {/* Heatmap cells */}
              {heatmapData.map((cell, idx) => (
                <g key={idx}>
                  <rect
                    x={cell.x * (380 / data.datasets.length)}
                    y={cell.y * (300 / data.methods.length)}
                    width={380 / data.datasets.length - 2}
                    height={300 / data.methods.length - 2}
                    fill={getColor(cell.value)}
                    stroke="white"
                    strokeWidth="1"
                  />
                  <text
                    x={cell.x * (380 / data.datasets.length) + (380 / data.datasets.length) / 2}
                    y={cell.y * (300 / data.methods.length) + (300 / data.methods.length) / 2}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize="10"
                    fill="white"
                  >
                    {cell.value.toFixed(3)}
                  </text>
                </g>
              ))}
              
              {/* Color scale legend */}
              <g transform="translate(400, 0)">
                <text x="0" y="-10" fontSize="12" fill="currentColor">Scale</text>
                {colors.map((color, idx) => (
                  <rect
                    key={idx}
                    x="0"
                    y={idx * (300 / colors.length)}
                    width="20"
                    height={300 / colors.length}
                    fill={color}
                  />
                ))}
                <text x="25" y="5" fontSize="10" fill="currentColor">
                  {maxValue.toFixed(3)}
                </text>
                <text x="25" y="295" fontSize="10" fill="currentColor">
                  {minValue.toFixed(3)}
                </text>
              </g>
            </g>
          </svg>
        </div>
      </CardContent>
    </Card>
  );
};

interface BoxPlotChartProps extends ChartProps {
  yAxisLabel?: string;
}

export const BoxPlotChart: React.FC<BoxPlotChartProps> = ({
  data,
  title,
  yAxisLabel,
  className,
  height = 400,
}) => {
  // Calculate box plot statistics
  const boxPlotData = data.labels.map((label: string, idx: number) => {
    const values = data.datasets[0].data[idx].sort((a: number, b: number) => a - b);
    const q1 = values[Math.floor(values.length * 0.25)];
    const median = values[Math.floor(values.length * 0.5)];
    const q3 = values[Math.floor(values.length * 0.75)];
    const min = values[0];
    const max = values[values.length - 1];
    
    return {
      name: label,
      min,
      q1,
      median,
      q3,
      max,
      values: [min, q1, median, q3, max]
    };
  });

  return (
    <Card className={className}>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={boxPlotData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis label={{ value: yAxisLabel, angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Bar dataKey="values" fill="none">
              {boxPlotData.map((entry, index) => (
                <Cell key={`cell-${index}`}>
                  <g>
                    {/* Box */}
                    <rect
                      x={index * 60 + 20}
                      y={entry.q3}
                      width={40}
                      height={entry.q1 - entry.q3}
                      fill="#3b82f6"
                      fillOpacity={0.5}
                      stroke="#3b82f6"
                    />
                    {/* Median line */}
                    <line
                      x1={index * 60 + 20}
                      x2={index * 60 + 60}
                      y1={entry.median}
                      y2={entry.median}
                      stroke="#1e40af"
                      strokeWidth={2}
                    />
                    {/* Whiskers */}
                    <line
                      x1={index * 60 + 40}
                      x2={index * 60 + 40}
                      y1={entry.min}
                      y2={entry.q1}
                      stroke="#3b82f6"
                      strokeDasharray="3 3"
                    />
                    <line
                      x1={index * 60 + 40}
                      x2={index * 60 + 40}
                      y1={entry.q3}
                      y2={entry.max}
                      stroke="#3b82f6"
                      strokeDasharray="3 3"
                    />
                  </g>
                </Cell>
              ))}
            </Bar>
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};