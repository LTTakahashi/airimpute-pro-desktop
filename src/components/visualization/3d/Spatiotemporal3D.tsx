import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  Html,
  Billboard,
  Text,
  Sphere,
  Plane
} from '@react-three/drei';
import * as THREE from 'three';
import { useSpring, animated } from '@react-spring/three';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Slider } from '@/components/ui/Slider';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/utils/cn';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Download, 
  Eye,
  EyeOff
} from 'lucide-react';
import chroma from 'chroma-js';

export interface Station {
  id: string;
  name: string;
  position: [number, number, number]; // [longitude, latitude, elevation]
  coordinates: { lat: number; lon: number; elevation: number };
}

export interface TimeSeriesData {
  stationId: string;
  timestamp: Date;
  value: number;
  uncertainty?: number;
  imputed?: boolean;
}

export interface InterpolationMethod {
  type: 'idw' | 'kriging' | 'spline' | 'nearest';
  params?: Record<string, any>;
}

export interface AnimationSettings {
  playing: boolean;
  speed: number; // frames per second
  loop: boolean;
  currentFrame: number;
}

interface Spatiotemporal3DProps {
  stations: Station[];
  data: TimeSeriesData[];
  timeRange: [Date, Date];
  currentTime?: Date;
  pollutant: string;
  interpolation?: InterpolationMethod;
  showUncertainty?: boolean;
  showGrid?: boolean;
  showLabels?: boolean;
  colorScale?: string[];
  valueRange?: [number, number];
  onTimeChange?: (time: Date) => void;
  onStationClick?: (station: Station) => void;
  className?: string;
}

// Color scales for different pollutants
const defaultColorScales = {
  PM25: ['#00ff00', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97', '#7e0023'],
  PM10: ['#00ff00', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97', '#7e0023'],
  NO2: ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000'],
  O3: ['#0000ff', '#00ff00', '#ffff00', '#ff7e00', '#ff0000'],
  default: ['#0000ff', '#00ff00', '#ffff00', '#ff0000']
};

// Helper function to normalize coordinates
function normalizeCoordinates(stations: Station[]): Station[] {
  if (stations.length === 0) return [];
  
  const lats = stations.map(s => s.coordinates.lat);
  const lons = stations.map(s => s.coordinates.lon);
  
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  
  const latRange = maxLat - minLat || 1;
  const lonRange = maxLon - minLon || 1;
  const scale = 10; // Scale to reasonable 3D units
  
  return stations.map(station => ({
    ...station,
    position: [
      ((station.coordinates.lon - minLon) / lonRange - 0.5) * scale,
      (station.coordinates.elevation || 0) / 100, // Convert to reasonable height
      ((station.coordinates.lat - minLat) / latRange - 0.5) * scale
    ]
  }));
}

// Station marker component
const StationMarker: React.FC<{
  station: Station;
  value: number | undefined;
  uncertainty: number | undefined;
  color: string;
  showLabel: boolean;
  showUncertainty: boolean;
  onClick?: () => void;
}> = ({ station, value, uncertainty, color, showLabel, showUncertainty, onClick }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  
  const { scale } = useSpring({
    scale: hovered ? 1.5 : 1,
    config: { tension: 300, friction: 20 }
  });
  
  useFrame((state) => {
    if (meshRef.current && value !== undefined) {
      // Subtle pulsing animation for imputed values
      const time = state.clock.getElapsedTime();
      meshRef.current.scale.y = 1 + Math.sin(time * 2) * 0.05;
    }
  });
  
  const height = value !== undefined ? (value / 100) * 3 : 0.1; // Scale height by value
  
  return (
    <group position={station.position}>
      {/* Base marker */}
      <animated.mesh
        ref={meshRef}
        scale={scale}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <cylinderGeometry args={[0.1, 0.1, height, 8]} />
        <meshStandardMaterial color={color} />
      </animated.mesh>
      
      {/* Uncertainty sphere */}
      {showUncertainty && uncertainty !== undefined && (
        <Sphere args={[uncertainty * 0.5, 16, 16]} position={[0, height / 2, 0]}>
          <meshStandardMaterial 
            color={color} 
            transparent 
            opacity={0.3}
            wireframe
          />
        </Sphere>
      )}
      
      {/* Label */}
      {showLabel && (
        <Billboard position={[0, height + 0.5, 0]}>
          <Text
            color="white"
            fontSize={0.3}
            anchorX="center"
            anchorY="middle"
            outlineWidth={0.02}
            outlineColor="black"
          >
            {station.name}
            {value !== undefined && `\n${value.toFixed(1)}`}
          </Text>
        </Billboard>
      )}
      
      {/* Hover tooltip */}
      {hovered && (
        <Html position={[0, height + 1, 0]} center>
          <Card className="p-2 text-xs">
            <div className="font-semibold">{station.name}</div>
            {value !== undefined && (
              <>
                <div>Value: {value.toFixed(2)}</div>
                {uncertainty !== undefined && (
                  <div>±{uncertainty.toFixed(2)}</div>
                )}
              </>
            )}
          </Card>
        </Html>
      )}
    </group>
  );
};

// Interpolated surface component
const InterpolatedSurface: React.FC<{
  stations: Station[];
  values: Map<string, number>;
  bounds: { min: [number, number, number]; max: [number, number, number] };
  resolution: number;
  colorScale: chroma.Scale;
  opacity?: number;
}> = ({ stations, values, bounds, resolution, colorScale, opacity = 0.7 }) => {
  const geometry = useMemo(() => {
    // Create a grid of points
    const geometry = new THREE.PlaneGeometry(
      bounds.max[0] - bounds.min[0],
      bounds.max[2] - bounds.min[2],
      resolution,
      resolution
    );
    
    // Simple IDW interpolation for demonstration
    const positions = geometry.attributes.position;
    const colors = new Float32Array(positions.count * 3);
    
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const z = positions.getZ(i);
      
      // IDW interpolation
      let weightedSum = 0;
      let weightSum = 0;
      
      stations.forEach(station => {
        const value = values.get(station.id);
        if (value !== undefined) {
          const dx = x - station.position[0];
          const dz = z - station.position[2];
          const distance = Math.sqrt(dx * dx + dz * dz) + 0.01; // Avoid division by zero
          const weight = 1 / (distance * distance);
          
          weightedSum += value * weight;
          weightSum += weight;
        }
      });
      
      const interpolatedValue = weightSum > 0 ? weightedSum / weightSum : 0;
      const color = colorScale(interpolatedValue / 100).rgb();
      
      colors[i * 3] = color[0] / 255;
      colors[i * 3 + 1] = color[1] / 255;
      colors[i * 3 + 2] = color[2] / 255;
      
      // Set height based on interpolated value
      positions.setY(i, interpolatedValue / 50);
    }
    
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();
    
    return geometry;
  }, [stations, values, bounds, resolution, colorScale]);
  
  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial 
        vertexColors
        transparent
        opacity={opacity}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

// Main 3D scene component
const Scene: React.FC<{
  stations: Station[];
  values: Map<string, number>;
  uncertainties: Map<string, number>;
  colorScale: chroma.Scale;
  valueRange: [number, number];
  showUncertainty: boolean;
  showGrid: boolean;
  showLabels: boolean;
  showInterpolation: boolean;
  onStationClick?: (station: Station) => void;
}> = ({ 
  stations, 
  values, 
  uncertainties,
  colorScale,
  valueRange,
  showUncertainty,
  showGrid,
  showLabels,
  showInterpolation,
  onStationClick
}) => {
  const { camera } = useThree();
  
  // Calculate bounds
  const bounds = useMemo(() => {
    if (stations.length === 0) {
      return { min: [-5, 0, -5] as [number, number, number], max: [5, 5, 5] as [number, number, number] };
    }
    
    const positions = stations.map(s => s.position);
    const xs = positions.map(p => p[0]);
    const ys = positions.map(p => p[1]);
    const zs = positions.map(p => p[2]);
    
    return {
      min: [Math.min(...xs) - 1, 0, Math.min(...zs) - 1] as [number, number, number],
      max: [Math.max(...xs) + 1, Math.max(...ys) + 2, Math.max(...zs) + 1] as [number, number, number]
    };
  }, [stations]);
  
  // Auto-adjust camera on first load
  useEffect(() => {
    const center = [
      (bounds.min[0] + bounds.max[0]) / 2,
      (bounds.min[1] + bounds.max[1]) / 2,
      (bounds.min[2] + bounds.max[2]) / 2
    ];
    
    camera.position.set(center[0] + 10, 10, center[2] + 10);
    camera.lookAt(center[0], center[1], center[2]);
  }, [bounds, camera]);
  
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
      <pointLight position={[-10, -10, -5]} intensity={0.5} />
      
      {/* Ground plane */}
      <Plane
        args={[bounds.max[0] - bounds.min[0] + 2, bounds.max[2] - bounds.min[2] + 2]}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, -0.01, 0]}
        receiveShadow
      >
        <meshStandardMaterial color="#f0f0f0" />
      </Plane>
      
      {/* Grid */}
      {showGrid && (
        <gridHelper 
          args={[
            Math.max(bounds.max[0] - bounds.min[0], bounds.max[2] - bounds.min[2]) + 2,
            20,
            0xcccccc,
            0xeeeeee
          ] as const} 
        />
      )}
      
      {/* Interpolated surface */}
      {showInterpolation && values.size > 0 && (
        <InterpolatedSurface
          stations={stations}
          values={values}
          bounds={bounds}
          resolution={30}
          colorScale={colorScale}
          opacity={0.7}
        />
      )}
      
      {/* Station markers */}
      {stations.map(station => {
        const value = values.get(station.id);
        const uncertainty = uncertainties.get(station.id);
        const normalizedValue = value !== undefined 
          ? (value - valueRange[0]) / (valueRange[1] - valueRange[0])
          : 0;
        const color = colorScale(normalizedValue).hex();
        
        return (
          <StationMarker
            key={station.id}
            station={station}
            value={value}
            uncertainty={uncertainty}
            color={color}
            showLabel={showLabels}
            showUncertainty={showUncertainty}
            onClick={() => onStationClick?.(station)}
          />
        );
      })}
      
      {/* Axes helper */}
      <axesHelper args={[5]} />
      
      {/* Controls */}
      <OrbitControls 
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={5}
        maxDistance={50}
      />
    </>
  );
};

export const Spatiotemporal3D: React.FC<Spatiotemporal3DProps> = ({
  stations,
  data,
  timeRange,
  currentTime,
  pollutant,
  showUncertainty: initialShowUncertainty = true,
  showGrid: initialShowGrid = true,
  showLabels: initialShowLabels = true,
  colorScale = (defaultColorScales as any)[pollutant] || defaultColorScales.default,
  valueRange = [0, 100],
  onTimeChange,
  onStationClick,
  className
}) => {
  const [animation, setAnimation] = useState<AnimationSettings>({
    playing: false,
    speed: 1,
    loop: true,
    currentFrame: 0
  });
  const [showInterpolation, setShowInterpolation] = useState(true);
  const [showControls, setShowControls] = useState(true);
  const [selectedTime, setSelectedTime] = useState(currentTime || timeRange[0]);
  const [showGrid, setShowGrid] = useState(initialShowGrid);
  const [showLabels, setShowLabels] = useState(initialShowLabels);
  const [showUncertainty] = useState(initialShowUncertainty);
  
  const animationRef = useRef<number>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Normalize station coordinates
  const normalizedStations = useMemo(() => normalizeCoordinates(stations), [stations]);
  
  // Get unique timestamps
  const timestamps = useMemo(() => {
    const uniqueTimes = new Set(data.map(d => d.timestamp.getTime()));
    return Array.from(uniqueTimes)
      .sort((a, b) => a - b)
      .map(t => new Date(t));
  }, [data]);
  
  // Get values for current time
  const currentValues = useMemo(() => {
    const values = new Map<string, number>();
    const uncertainties = new Map<string, number>();
    
    data
      .filter(d => d.timestamp.getTime() === selectedTime.getTime())
      .forEach(d => {
        values.set(d.stationId, d.value);
        if (d.uncertainty !== undefined) {
          uncertainties.set(d.stationId, d.uncertainty);
        }
      });
    
    return { values, uncertainties };
  }, [data, selectedTime]);
  
  // Create color scale
  const chromaColorScale = useMemo(() => {
    return chroma.scale(colorScale).domain(valueRange);
  }, [colorScale, valueRange]);
  
  // Animation loop
  useEffect(() => {
    if (animation.playing && timestamps.length > 0) {
      const frameDelay = 1000 / animation.speed;
      
      animationRef.current = window.setTimeout(() => {
        const nextFrame = (animation.currentFrame + 1) % timestamps.length;
        const nextTime = timestamps[nextFrame];
        
        setAnimation(prev => ({ ...prev, currentFrame: nextFrame }));
        setSelectedTime(nextTime);
        onTimeChange?.(nextTime);
        
        if (!animation.loop && nextFrame === 0) {
          setAnimation(prev => ({ ...prev, playing: false }));
        }
      }, frameDelay);
    }
    
    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, [animation, timestamps, onTimeChange]);
  
  // Handle time slider change
  const handleTimeChange = (value: number[]) => {
    const index = value[0];
    const time = timestamps[index];
    setSelectedTime(time);
    setAnimation(prev => ({ ...prev, currentFrame: index }));
    onTimeChange?.(time);
  };
  
  // Export functionality
  const handleExport = async () => {
    if (!canvasRef.current) return;
    
    // Convert canvas to blob
    canvasRef.current.toBlob((blob) => {
      if (!blob) return;
      
      // Create download link
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `spatiotemporal_${pollutant}_${selectedTime.toISOString()}.png`;
      a.click();
      URL.revokeObjectURL(url);
    });
  };
  
  return (
    <div className={cn('relative h-full w-full', className)}>
      {/* 3D Canvas */}
      <Canvas
        ref={canvasRef}
        shadows
        camera={{ 
          position: [15, 15, 15],
          fov: 45,
          near: 0.1,
          far: 1000
        }}
        className="bg-gray-100 dark:bg-gray-900"
      >
        <Scene
          stations={normalizedStations}
          values={currentValues.values}
          uncertainties={currentValues.uncertainties}
          colorScale={chromaColorScale}
          valueRange={valueRange}
          showUncertainty={showUncertainty}
          showGrid={showGrid}
          showLabels={showLabels}
          showInterpolation={showInterpolation}
          onStationClick={onStationClick}
        />
      </Canvas>
      
      {/* Controls Overlay */}
      {showControls && (
        <div className="absolute top-4 left-4 right-4 pointer-events-none">
          <div className="flex justify-between items-start pointer-events-auto">
            {/* Left controls */}
            <Card className="p-4 bg-white/90 dark:bg-gray-800/90 backdrop-blur">
              <div className="space-y-3">
                <div>
                  <h3 className="font-semibold mb-2">3D Spatiotemporal View</h3>
                  <Badge variant="secondary">{pollutant}</Badge>
                </div>
                
                <div className="space-y-2">
                  <label className="text-sm font-medium">Display Options</label>
                  <div className="space-y-1">
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={showInterpolation}
                        onChange={(e) => setShowInterpolation(e.target.checked)}
                        className="rounded"
                      />
                      Show Interpolation
                    </label>
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={showGrid}
                        onChange={(e) => setShowGrid(e.target.checked)}
                        className="rounded"
                      />
                      Show Grid
                    </label>
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={showLabels}
                        onChange={(e) => setShowLabels(e.target.checked)}
                        className="rounded"
                      />
                      Show Labels
                    </label>
                  </div>
                </div>
              </div>
            </Card>
            
            {/* Right controls */}
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowControls(false)}
                className="bg-white/90 dark:bg-gray-800/90"
              >
                <EyeOff className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleExport}
                className="bg-white/90 dark:bg-gray-800/90"
              >
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      )}
      
      {/* Show controls button when hidden */}
      {!showControls && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowControls(true)}
          className="absolute top-4 left-4"
        >
          <Eye className="h-4 w-4" />
        </Button>
      )}
      
      {/* Timeline Controls */}
      <div className="absolute bottom-4 left-4 right-4">
        <Card className="p-4 bg-white/90 dark:bg-gray-800/90 backdrop-blur">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Button
                  variant={animation.playing ? 'outline' : 'primary'}
                  size="sm"
                  onClick={() => setAnimation(prev => ({ ...prev, playing: !prev.playing }))}
                >
                  {animation.playing ? (
                    <Pause className="h-4 w-4" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setAnimation(prev => ({ ...prev, currentFrame: 0 }));
                    setSelectedTime(timestamps[0]);
                  }}
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
                
                <div className="text-sm">
                  <span className="font-medium">Time:</span>{' '}
                  {selectedTime.toLocaleString()}
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <label className="text-sm">
                  Speed: {animation.speed}x
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="4"
                  step="0.5"
                  value={animation.speed}
                  onChange={(e) => setAnimation(prev => ({ 
                    ...prev, 
                    speed: parseFloat(e.target.value) 
                  }))}
                  className="w-24"
                />
              </div>
            </div>
            
            {/* Time slider */}
            <div className="relative">
              <Slider
                value={[animation.currentFrame]}
                onValueChange={handleTimeChange}
                max={timestamps.length - 1}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>{timeRange[0].toLocaleDateString()}</span>
                <span>{timeRange[1].toLocaleDateString()}</span>
              </div>
            </div>
          </div>
        </Card>
      </div>
      
      {/* Color scale legend */}
      <div className="absolute top-20 right-4">
        <Card className="p-3 bg-white/90 dark:bg-gray-800/90 backdrop-blur">
          <div className="text-sm font-medium mb-2">Value Scale</div>
          <div className="w-32 h-4 relative rounded overflow-hidden">
            <div 
              className="absolute inset-0"
              style={{
                background: `linear-gradient(to right, ${colorScale.join(', ')})`
              }}
            />
          </div>
          <div className="flex justify-between text-xs mt-1">
            <span>{valueRange[0]}</span>
            <span>{valueRange[1]}</span>
          </div>
        </Card>
      </div>
    </div>
  );
};