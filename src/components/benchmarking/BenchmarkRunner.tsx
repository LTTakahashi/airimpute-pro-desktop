import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';

export const BenchmarkRunner: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Benchmark Runner</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">
          Benchmark runner component - integrated into main dashboard
        </p>
      </CardContent>
    </Card>
  );
};