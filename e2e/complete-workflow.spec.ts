import { test, expect, Page } from '@playwright/test';
import { promises as fs } from 'fs';
import path from 'path';

// Test data paths
const TEST_DATA_DIR = path.join(__dirname, 'test-data');
const SAMPLE_CSV = path.join(TEST_DATA_DIR, 'air_quality_sample.csv');

// Helper functions
async function waitForNotification(page: Page, text: string) {
  await page.locator('.notification', { hasText: text }).waitFor({ state: 'visible' });
}

async function uploadFile(page: Page, filePath: string) {
  const fileChooserPromise = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: /import data/i }).click();
  const fileChooser = await fileChooserPromise;
  await fileChooser.setFiles(filePath);
}

test.describe('AirImpute Pro - Complete Workflow', () => {
  test.beforeAll(async () => {
    // Create test data directory
    await fs.mkdir(TEST_DATA_DIR, { recursive: true });
    
    // Create sample CSV file
    const csvContent = `timestamp,pm25,pm10,no2,o3,co,so2
2024-01-01 00:00:00,25.5,45.2,38.1,42.3,0.8,12.5
2024-01-01 01:00:00,26.1,,39.2,41.8,0.9,13.1
2024-01-01 02:00:00,24.8,44.1,,40.5,0.7,
2024-01-01 03:00:00,23.2,42.5,36.8,,0.6,11.8
2024-01-01 04:00:00,,41.8,35.4,38.2,,10.9
2024-01-01 05:00:00,22.1,40.2,34.1,37.5,0.5,10.2`;
    
    await fs.writeFile(SAMPLE_CSV, csvContent);
  });

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('Complete data analysis workflow', async ({ page }) => {
    // Step 1: Navigate to Data Import
    await page.getByRole('link', { name: /data import/i }).click();
    await expect(page).toHaveURL(/.*\/data-import/);

    // Step 2: Upload CSV file
    await uploadFile(page, SAMPLE_CSV);
    
    // Wait for file processing
    await expect(page.getByText(/processing file/i)).toBeVisible();
    await expect(page.getByText(/processing file/i)).toBeHidden({ timeout: 10000 });

    // Verify data preview
    await expect(page.getByText('air_quality_sample.csv')).toBeVisible();
    await expect(page.getByText(/6 rows/i)).toBeVisible();
    await expect(page.getByText(/6 columns/i)).toBeVisible();

    // Step 3: Configure import options
    await page.getByLabel(/parse dates/i).check();
    await page.getByLabel(/date column/i).selectOption('timestamp');
    
    // Import data
    await page.getByRole('button', { name: /import/i }).click();
    await waitForNotification(page, 'Dataset imported successfully');

    // Step 4: Navigate to Imputation
    await page.getByRole('link', { name: /imputation/i }).click();
    
    // Select dataset
    await page.getByLabel(/select dataset/i).selectOption('air_quality_sample');
    
    // Wait for data to load
    await expect(page.getByText(/missing values: 7/i)).toBeVisible();
    await expect(page.getByText(/19.44%/i)).toBeVisible();

    // Select imputation method
    await page.getByRole('tab', { name: /statistical/i }).click();
    await page.getByText('Kalman Filter').click();

    // Configure parameters
    await page.getByLabel(/model order/i).fill('2');
    await page.getByLabel(/process noise/i).fill('0.01');

    // Run imputation
    await page.getByRole('button', { name: /run imputation/i }).click();

    // Wait for progress
    await expect(page.getByRole('progressbar')).toBeVisible();
    await expect(page.getByText(/imputation completed/i)).toBeVisible({ timeout: 30000 });

    // Verify results
    await expect(page.getByText(/rmse:/i)).toBeVisible();
    await expect(page.getByText(/mae:/i)).toBeVisible();
    await expect(page.getByText(/r²:/i)).toBeVisible();

    // Step 5: Visualize results
    await page.getByRole('link', { name: /visualization/i }).click();
    
    // Create time series plot
    await page.getByRole('button', { name: /new plot/i }).click();
    await page.getByLabel(/plot type/i).selectOption('time_series');
    
    // Select variables
    await page.getByLabel('PM2.5').check();
    await page.getByLabel('NO2').check();
    
    // Generate plot
    await page.getByRole('button', { name: /generate/i }).click();
    
    // Wait for plot to render
    await expect(page.locator('.plot-container canvas')).toBeVisible({ timeout: 10000 });

    // Step 6: Export results
    await page.getByRole('link', { name: /export/i }).click();
    
    // Select export format
    await page.getByRole('radio', { name: /excel/i }).check();
    
    // Configure export options
    await page.getByLabel(/include metadata/i).check();
    await page.getByLabel(/include plots/i).check();
    
    // Export
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: /export/i }).click();
    const download = await downloadPromise;
    
    // Verify download
    expect(download.suggestedFilename()).toContain('air_quality_sample');
    expect(download.suggestedFilename()).toEndWith('.xlsx');
  });

  test('Benchmarking workflow', async ({ page }) => {
    // Navigate to benchmarking
    await page.getByRole('link', { name: /benchmarking/i }).click();
    
    // Import test dataset first
    await page.getByRole('button', { name: /import dataset/i }).click();
    await uploadFile(page, SAMPLE_CSV);
    await waitForNotification(page, 'Dataset imported');

    // Select dataset for benchmarking
    await page.getByText('air_quality_sample').click();
    
    // Select methods to benchmark
    await page.getByRole('tab', { name: /all methods/i }).click();
    await page.getByRole('button', { name: /select all simple/i }).click();
    await page.getByRole('button', { name: /select all statistical/i }).click();

    // Configure benchmark settings
    await page.getByLabel(/cross-validation folds/i).fill('5');
    await page.getByLabel(/use gpu/i).check();

    // Run benchmark
    await page.getByRole('button', { name: /run benchmark/i }).click();

    // Wait for completion
    await expect(page.getByText(/benchmark in progress/i)).toBeVisible();
    await expect(page.getByText(/benchmark completed/i)).toBeVisible({ timeout: 60000 });

    // Verify results table
    const resultsTable = page.locator('table.benchmark-results');
    await expect(resultsTable).toBeVisible();
    
    // Check that we have results for multiple methods
    const rows = resultsTable.locator('tbody tr');
    await expect(rows).toHaveCount({ min: 2 });

    // Verify best method is highlighted
    await expect(page.locator('.best-method')).toBeVisible();

    // Export benchmark report
    await page.getByRole('button', { name: /export report/i }).click();
    await page.getByRole('menuitem', { name: /latex/i }).click();
    
    const reportDownload = page.waitForEvent('download');
    await page.getByRole('button', { name: /generate/i }).click();
    await reportDownload;
  });

  test('Error handling and recovery', async ({ page }) => {
    // Try to import invalid file
    await page.getByRole('link', { name: /data import/i }).click();
    
    // Create invalid CSV
    const invalidCsv = path.join(TEST_DATA_DIR, 'invalid.csv');
    await fs.writeFile(invalidCsv, 'invalid,data\n1,2,3,4,5'); // Mismatched columns
    
    await uploadFile(page, invalidCsv);
    
    // Should show error
    await expect(page.getByText(/error.*invalid csv format/i)).toBeVisible();
    
    // Should allow retry
    await expect(page.getByRole('button', { name: /try again/i })).toBeVisible();

    // Test imputation with insufficient data
    await page.getByRole('link', { name: /imputation/i }).click();
    
    // Create dataset with too many missing values
    const sparseData = `timestamp,value
2024-01-01 00:00:00,10
2024-01-01 01:00:00,
2024-01-01 02:00:00,
2024-01-01 03:00:00,
2024-01-01 04:00:00,20`;
    
    const sparseCsv = path.join(TEST_DATA_DIR, 'sparse.csv');
    await fs.writeFile(sparseCsv, sparseData);
    
    // Import sparse data
    await page.getByRole('button', { name: /import new/i }).click();
    await uploadFile(page, sparseCsv);
    await waitForNotification(page, 'Dataset imported');
    
    // Try complex imputation method
    await page.getByLabel(/select dataset/i).selectOption('sparse');
    await page.getByRole('tab', { name: /machine learning/i }).click();
    await page.getByText('Random Forest').click();
    await page.getByRole('button', { name: /run imputation/i }).click();
    
    // Should show warning about insufficient data
    await expect(page.getByText(/warning.*insufficient data/i)).toBeVisible();
    
    // Should suggest simpler method
    await expect(page.getByText(/try.*simpler method/i)).toBeVisible();
  });

  test('Performance with large dataset', async ({ page }) => {
    // Create large dataset (10k rows)
    const largeData = ['timestamp,pm25,pm10,no2,o3,co,so2'];
    const baseDate = new Date('2024-01-01');
    
    for (let i = 0; i < 10000; i++) {
      const date = new Date(baseDate.getTime() + i * 3600000); // Hourly data
      const row = [
        date.toISOString().replace('T', ' ').slice(0, -5),
        Math.random() > 0.1 ? (20 + Math.random() * 30).toFixed(1) : '',
        Math.random() > 0.1 ? (40 + Math.random() * 40).toFixed(1) : '',
        Math.random() > 0.1 ? (30 + Math.random() * 20).toFixed(1) : '',
        Math.random() > 0.1 ? (35 + Math.random() * 25).toFixed(1) : '',
        Math.random() > 0.1 ? (0.5 + Math.random() * 0.5).toFixed(2) : '',
        Math.random() > 0.1 ? (10 + Math.random() * 10).toFixed(1) : '',
      ].join(',');
      largeData.push(row);
    }
    
    const largeCsv = path.join(TEST_DATA_DIR, 'large_dataset.csv');
    await fs.writeFile(largeCsv, largeData.join('\n'));
    
    // Import large dataset
    await page.getByRole('link', { name: /data import/i }).click();
    await uploadFile(page, largeCsv);
    
    // Time the import
    const startTime = Date.now();
    await waitForNotification(page, 'Dataset imported successfully');
    const importTime = Date.now() - startTime;
    
    // Should complete within reasonable time (< 30s)
    expect(importTime).toBeLessThan(30000);
    
    // Verify dataset info
    await expect(page.getByText(/10,000 rows/i)).toBeVisible();
    
    // Test visualization performance
    await page.getByRole('link', { name: /visualization/i }).click();
    await page.getByRole('button', { name: /new plot/i }).click();
    await page.getByLabel(/plot type/i).selectOption('time_series');
    await page.getByLabel('PM2.5').check();
    
    // Time the plot generation
    const plotStart = Date.now();
    await page.getByRole('button', { name: /generate/i }).click();
    await expect(page.locator('.plot-container canvas')).toBeVisible();
    const plotTime = Date.now() - plotStart;
    
    // Should render within reasonable time (< 5s)
    expect(plotTime).toBeLessThan(5000);
  });

  test('Keyboard navigation and accessibility', async ({ page }) => {
    // Test keyboard navigation
    await page.keyboard.press('Tab');
    await expect(page.locator(':focus')).toHaveAttribute('aria-label');
    
    // Navigate to data import using keyboard
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter');
    
    await expect(page).toHaveURL(/.*\/data-import/);
    
    // Test screen reader announcements
    await page.getByRole('button', { name: /import data/i }).focus();
    const ariaLabel = await page.locator(':focus').getAttribute('aria-label');
    expect(ariaLabel).toBeTruthy();
    
    // Test high contrast mode
    await page.emulateMedia({ colorScheme: 'dark' });
    await expect(page.locator('body')).toHaveClass(/dark/);
    
    // Verify contrast ratios (would need axe-core for full testing)
    const backgroundColor = await page.locator('body').evaluate(
      el => window.getComputedStyle(el).backgroundColor
    );
    const textColor = await page.locator('h1').evaluate(
      el => window.getComputedStyle(el).color
    );
    
    // Basic check that they're different
    expect(backgroundColor).not.toBe(textColor);
  });

  test.afterAll(async () => {
    // Cleanup test data
    await fs.rm(TEST_DATA_DIR, { recursive: true, force: true });
  });
});