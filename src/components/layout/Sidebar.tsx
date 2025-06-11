import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Home,
  Upload,
  Activity,
  BarChart3,
  Image,
  Download,
  FileText,
  Settings,
  HelpCircle,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { useStore } from '@/store';

interface NavItem {
  path: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  { path: '/', label: 'Dashboard', icon: Home },
  { path: '/import', label: 'Import Data', icon: Upload },
  { path: '/imputation', label: 'Imputation', icon: Activity },
  { path: '/analysis', label: 'Analysis', icon: BarChart3 },
  { path: '/visualization', label: 'Visualization', icon: Image },
  { path: '/export', label: 'Export', icon: Download },
  { path: '/publication', label: 'Publication', icon: FileText },
  { path: '/settings', label: 'Settings', icon: Settings },
  { path: '/help', label: 'Help', icon: HelpCircle }
];

const Sidebar: React.FC = () => {
  const { sidebarCollapsed, setSidebarCollapsed } = useStore();

  return (
    <aside
      className={`
        bg-gray-900 text-white transition-all duration-300 ease-in-out
        ${sidebarCollapsed ? 'w-16' : 'w-64'}
        min-h-screen flex flex-col
      `}
    >
      {/* Logo */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between">
          {!sidebarCollapsed && (
            <h1 className="text-xl font-bold">AirImpute Pro</h1>
          )}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="p-1 rounded hover:bg-gray-800 transition-colors"
          >
            {sidebarCollapsed ? (
              <ChevronRight className="w-5 h-5" />
            ) : (
              <ChevronLeft className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map(item => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) => `
                  flex items-center px-3 py-2 rounded-lg transition-colors
                  ${isActive
                    ? 'bg-blue-600 text-white'
                    : 'hover:bg-gray-800 text-gray-300 hover:text-white'}
                `}
                title={sidebarCollapsed ? item.label : undefined}
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                {!sidebarCollapsed && (
                  <span className="ml-3">{item.label}</span>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Version */}
      {!sidebarCollapsed && (
        <div className="p-4 border-t border-gray-800 text-xs text-gray-500">
          Version 1.0.0
        </div>
      )}
    </aside>
  );
};

export default Sidebar;