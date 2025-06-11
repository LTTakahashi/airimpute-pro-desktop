import { createBrowserRouter } from 'react-router-dom'
import { App } from './App'
import { Dashboard } from './pages/Dashboard'
import DataImport from './pages/DataImport'
import Imputation from './pages/ImputationV2' // Updated to use V2
import Analysis from './pages/Analysis'
import Visualization from './pages/Visualization'
import Export from './pages/Export'
import Publication from './pages/Publication'
import Settings from './pages/Settings'
import Help from './pages/Help'
import NotFound from './pages/NotFound'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    errorElement: <NotFound />,
    children: [
      {
        index: true,
        element: <Dashboard />,
      },
      {
        path: 'import',
        element: <DataImport />,
      },
      {
        path: 'imputation',
        element: <Imputation />,
      },
      {
        path: 'analysis',
        element: <Analysis />,
      },
      {
        path: 'visualization',
        element: <Visualization />,
      },
      {
        path: 'export',
        element: <Export />,
      },
      {
        path: 'publication',
        element: <Publication />,
      },
      {
        path: 'settings',
        element: <Settings />,
      },
      {
        path: 'help',
        element: <Help />,
      },
    ],
  },
])