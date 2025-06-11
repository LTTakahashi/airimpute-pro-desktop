import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import { Checkbox } from '../ui/Checkbox';
import { Progress } from '../ui/Progress';
import { 
  Zap, 
  Brain, 
  TrendingUp, 
  BarChart2, 
  GitBranch,
  Cpu,
  Timer,
  Info,
  Settings,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { cn } from '@/utils/cn';

interface Method {
  name: string;
  category: string;
  hasGPUSupport: boolean;
  parameters: Record<string, any>;
  description: string;
}

interface MethodComparisonProps {
  methods: Method[];
  selectedMethods: string[];
  onSelectionChange: (methods: string[]) => void;
  loading?: boolean;
  className?: string;
}

const categoryIcons: Record<string, React.ElementType> = {
  'statistical': BarChart2,
  'machine_learning': Brain,
  'deep_learning': GitBranch,
  'ensemble': TrendingUp,
  'interpolation': Timer
};

const categoryColors: Record<string, string> = {
  'statistical': 'secondary',
  'machine_learning': 'default',
  'deep_learning': 'destructive',
  'ensemble': 'success',
  'interpolation': 'outline'
};

export const MethodComparison: React.FC<MethodComparisonProps> = ({
  methods,
  selectedMethods,
  onSelectionChange,
  loading = false,
  className
}) => {
  const [expandedMethods, setExpandedMethods] = useState<string[]>([]);
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);

  const toggleMethod = (methodName: string) => {
    if (selectedMethods.includes(methodName)) {
      onSelectionChange(selectedMethods.filter(m => m !== methodName));
    } else {
      onSelectionChange([...selectedMethods, methodName]);
    }
  };

  const toggleExpanded = (methodName: string) => {
    setExpandedMethods(prev => 
      prev.includes(methodName) 
        ? prev.filter(m => m !== methodName)
        : [...prev, methodName]
    );
  };

  const selectCategory = (category: string) => {
    const categoryMethods = methods
      .filter(m => m.category === category)
      .map(m => m.name);
    
    const allSelected = categoryMethods.every(m => selectedMethods.includes(m));
    
    if (allSelected) {
      onSelectionChange(selectedMethods.filter(m => !categoryMethods.includes(m)));
    } else {
      onSelectionChange([...new Set([...selectedMethods, ...categoryMethods])]);
    }
  };

  const getMethodComplexity = (method: Method): 'low' | 'medium' | 'high' => {
    const paramCount = Object.keys(method.parameters).length;
    if (paramCount <= 2) return 'low';
    if (paramCount <= 5) return 'medium';
    return 'high';
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low': return 'text-green-600';
      case 'medium': return 'text-yellow-600';
      case 'high': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  // Group methods by category
  const groupedMethods = methods.reduce((acc, method) => {
    if (!acc[method.category]) {
      acc[method.category] = [];
    }
    acc[method.category].push(method);
    return acc;
  }, {} as Record<string, Method[]>);

  const filteredGroups = categoryFilter 
    ? { [categoryFilter]: groupedMethods[categoryFilter] || [] }
    : groupedMethods;

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="py-8">
          <div className="flex flex-col items-center justify-center space-y-4">
            <Brain className="w-8 h-8 animate-pulse text-muted-foreground" />
            <p className="text-sm text-muted-foreground">Loading methods...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={cn("space-y-4", className)}>
      {/* Category Filter */}
      <div className="flex items-center gap-2 flex-wrap">
        <Button
          variant={categoryFilter === null ? "default" : "outline"}
          size="sm"
          onClick={() => setCategoryFilter(null)}
        >
          All Categories
        </Button>
        {Object.keys(groupedMethods).map(category => {
          const Icon = categoryIcons[category] || BarChart2;
          const methodCount = groupedMethods[category].length;
          const selectedCount = groupedMethods[category].filter(m => 
            selectedMethods.includes(m.name)
          ).length;
          
          return (
            <Button
              key={category}
              variant={categoryFilter === category ? "default" : "outline"}
              size="sm"
              onClick={() => setCategoryFilter(category)}
              className="gap-1"
            >
              <Icon className="w-3 h-3" />
              {category.replace('_', ' ')}
              <Badge variant="secondary" className="ml-1 h-5 px-1">
                {selectedCount}/{methodCount}
              </Badge>
            </Button>
          );
        })}
      </div>

      {/* Methods by Category */}
      <div className="space-y-4">
        {Object.entries(filteredGroups).map(([category, categoryMethods]) => {
          const Icon = categoryIcons[category] || BarChart2;
          const allSelected = categoryMethods.every(m => selectedMethods.includes(m.name));
          const someSelected = categoryMethods.some(m => selectedMethods.includes(m.name));
          
          return (
            <Card key={category}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Icon className="w-5 h-5" />
                    {category.replace('_', ' ').charAt(0).toUpperCase() + category.slice(1).replace('_', ' ')}
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">
                      {categoryMethods.filter(m => selectedMethods.includes(m.name)).length} / {categoryMethods.length} selected
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => selectCategory(category)}
                    >
                      {allSelected ? 'Deselect All' : 'Select All'}
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                {categoryMethods.map(method => {
                  const isSelected = selectedMethods.includes(method.name);
                  const isExpanded = expandedMethods.includes(method.name);
                  const complexity = getMethodComplexity(method);
                  
                  return (
                    <div
                      key={method.name}
                      className={cn(
                        "border rounded-lg p-3 transition-all",
                        isSelected && "border-primary bg-primary/5"
                      )}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3 flex-1">
                          <Checkbox
                            checked={isSelected}
                            onCheckedChange={() => toggleMethod(method.name)}
                            className="mt-0.5"
                          />
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <h4 className="font-medium">{method.name}</h4>
                              {method.hasGPUSupport && (
                                <Badge variant="outline" className="gap-1 h-5">
                                  <Zap className="w-3 h-3" />
                                  GPU
                                </Badge>
                              )}
                              <Badge 
                                variant="secondary" 
                                className={cn("h-5", getComplexityColor(complexity))}
                              >
                                {complexity} complexity
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mt-1">
                              {method.description}
                            </p>
                            
                            {isExpanded && (
                              <div className="mt-3 space-y-2">
                                <div className="text-sm">
                                  <h5 className="font-medium mb-1">Parameters:</h5>
                                  <div className="grid grid-cols-2 gap-2">
                                    {Object.entries(method.parameters).map(([param, value]) => (
                                      <div key={param} className="flex justify-between text-xs">
                                        <span className="text-muted-foreground">{param}:</span>
                                        <span className="font-mono">{JSON.stringify(value)}</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                                
                                <div className="flex gap-2">
                                  <Button size="sm" variant="outline" className="text-xs">
                                    <Settings className="w-3 h-3 mr-1" />
                                    Configure
                                  </Button>
                                  <Button size="sm" variant="outline" className="text-xs">
                                    <Info className="w-3 h-3 mr-1" />
                                    Documentation
                                  </Button>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                        
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleExpanded(method.name)}
                          className="ml-2"
                        >
                          {isExpanded ? (
                            <ChevronUp className="w-4 h-4" />
                          ) : (
                            <ChevronDown className="w-4 h-4" />
                          )}
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Summary Stats */}
      <Card>
        <CardContent className="pt-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-2xl font-semibold">{selectedMethods.length}</p>
              <p className="text-sm text-muted-foreground">Selected Methods</p>
            </div>
            <div>
              <p className="text-2xl font-semibold">
                {methods.filter(m => m.hasGPUSupport && selectedMethods.includes(m.name)).length}
              </p>
              <p className="text-sm text-muted-foreground">GPU Accelerated</p>
            </div>
            <div>
              <p className="text-2xl font-semibold">
                {Object.keys(filteredGroups).length}
              </p>
              <p className="text-sm text-muted-foreground">Categories</p>
            </div>
            <div>
              <p className="text-2xl font-semibold">
                {selectedMethods.reduce((acc, methodName) => {
                  const method = methods.find(m => m.name === methodName);
                  return acc + (method ? Object.keys(method.parameters).length : 0);
                }, 0)}
              </p>
              <p className="text-sm text-muted-foreground">Total Parameters</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};