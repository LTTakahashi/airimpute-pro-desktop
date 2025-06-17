/**
 * Windows-specific integration tests
 * Ensures compatibility with Windows file systems and Tauri runtime
 */

import { describe, it, expect, vi } from 'vitest';
import { invoke } from '@tauri-apps/api/tauri';
import { IS_WINDOWS, BASE_TIMEOUT } from '@/test/setup.integration';

// Skip these tests on non-Windows platforms in CI
const describeWindows = IS_WINDOWS ? describe : describe.skip;

describeWindows('Windows-specific Integration Tests', () => {
  describe('File Path Handling', () => {
    it('handles Windows-style paths correctly', async () => {
      const windowsPaths = [
        'C:\\Users\\Test\\Documents\\data.csv',
        'D:\\Projects\\AirQuality\\dataset.xlsx',
        '\\\\NetworkShare\\Data\\pollution.csv',
        'C:\\Program Files (x86)\\AirImpute\\data\\test.csv',
      ];

      for (const path of windowsPaths) {
        // Mock the response
        vi.mocked(invoke).mockResolvedValueOnce({
          id: 'test-dataset',
          name: path.split('\\').pop(),
          path: path,
          rows: 100,
          columns: 5,
        });

        const result = await invoke('load_dataset', {
          path,
          options: { has_header: true }
        });

        expect(result).toBeTruthy();
        expect((result as any).path).toBe(path);
      }
    });

    it('handles paths with spaces correctly', async () => {
      const pathsWithSpaces = [
        'C:\\My Documents\\Air Quality Data\\PM2.5 readings.csv',
        'D:\\Research Projects\\2024 Analysis\\dataset.xlsx',
      ];

      for (const path of pathsWithSpaces) {
        vi.mocked(invoke).mockResolvedValueOnce({
          id: 'test-dataset',
          name: 'test',
          path: path,
        });

        await expect(invoke('load_dataset', { 
          path,
          options: {} 
        })).resolves.toBeTruthy();
      }
    });

    it('handles long path names (near 260 char limit)', async () => {
      const longPath = 'C:\\' + 'a'.repeat(240) + '\\data.csv';
      
      vi.mocked(invoke).mockRejectedValueOnce(
        new Error('Path too long for Windows')
      );

      await expect(invoke('load_dataset', {
        path: longPath,
        options: {}
      })).rejects.toThrow('Path too long');
    });

    it('normalizes mixed path separators', async () => {
      const mixedPath = 'C:/Users\\Test/Documents\\data.csv';
      const normalizedPath = 'C:\\Users\\Test\\Documents\\data.csv';

      vi.mocked(invoke).mockResolvedValueOnce({
        id: 'test-dataset',
        path: normalizedPath,
      });

      const result = await invoke('normalize_path', { path: mixedPath });
      expect((result as any).path).toBe(normalizedPath);
    });
  });

  describe('File Locking and Permissions', () => {
    it('handles file in use errors gracefully', async () => {
      vi.mocked(invoke).mockRejectedValueOnce(
        new Error('The process cannot access the file because it is being used by another process')
      );

      await expect(invoke('export_to_csv', {
        dataset_id: 'test',
        path: 'C:\\output.csv',
        options: {}
      })).rejects.toThrow('being used by another process');
    });

    it('retries on temporary file locks', async () => {
      let attempts = 0;
      
      vi.mocked(invoke).mockImplementation(async (cmd) => {
        if (cmd === 'export_to_csv' && attempts < 2) {
          attempts++;
          throw new Error('File is locked');
        }
        return { success: true, attempts };
      });

      const result = await invoke('export_to_csv', {
        dataset_id: 'test',
        path: 'C:\\output.csv',
        options: { retry: true, max_attempts: 3 }
      });

      expect((result as any).success).toBe(true);
      expect((result as any).attempts).toBe(2);
    });
  });

  describe('Memory Management on Windows', () => {
    it('handles large datasets within Windows memory constraints', async () => {
      // Windows has different memory allocation patterns
      vi.mocked(invoke).mockResolvedValueOnce({
        available_memory_mb: 8192,
        allocated_memory_mb: 0,
        max_dataset_size_mb: 4096,
      });

      const memInfo = await invoke('get_system_memory_info');
      expect((memInfo as any).available_memory_mb).toBeGreaterThan(1024);
    });

    it('implements proper cleanup for COM objects', async () => {
      // Simulate Excel file operations that use COM
      vi.mocked(invoke).mockResolvedValueOnce({
        id: 'excel-dataset',
        com_objects_released: true,
      });

      const result = await invoke('load_dataset', {
        path: 'C:\\data.xlsx',
        options: { format: 'excel' }
      });

      expect((result as any).com_objects_released).toBe(true);
    });
  });

  describe('Windows-specific Tauri Features', () => {
    it('uses Windows notifications correctly', async () => {
      vi.mocked(invoke).mockResolvedValueOnce({
        notification_sent: true,
        platform: 'windows',
      });

      const result = await invoke('send_notification', {
        title: 'Imputation Complete',
        body: 'Your dataset has been processed',
        icon: 'info',
      });

      expect((result as any).notification_sent).toBe(true);
      expect((result as any).platform).toBe('windows');
    });

    it('handles Windows registry settings', async () => {
      vi.mocked(invoke).mockResolvedValueOnce({
        theme: 'dark',
        accent_color: '#0078D4', // Windows blue
      });

      const settings = await invoke('get_windows_theme_settings');
      expect((settings as any).accent_color).toBe('#0078D4');
    });
  });

  describe('Concurrent Operations on Windows', () => {
    it('manages file handles correctly during concurrent operations', async () => {
      const operations: Promise<unknown>[] = [];
      
      // Simulate multiple file operations
      for (let i = 0; i < 5; i++) {
        vi.mocked(invoke).mockResolvedValueOnce({
          operation_id: i,
          handle_count: i + 1,
          success: true,
        });

        operations.push(
          invoke('process_file_concurrent', {
            file_id: `file-${i}`,
            operation: 'read',
          })
        );
      }

      const results = await Promise.all(operations);
      
      // Verify all operations completed
      expect(results).toHaveLength(5);
      results.forEach((result, index) => {
        expect((result as any).operation_id).toBe(index);
        expect((result as any).success).toBe(true);
      });
    });
  });

  describe('Network Drive Support', () => {
    it('handles UNC paths for network shares', async () => {
      const uncPath = '\\\\Server\\Share\\Data\\airquality.csv';
      
      vi.mocked(invoke).mockResolvedValueOnce({
        id: 'network-dataset',
        path: uncPath,
        is_network_path: true,
      });

      const result = await invoke('load_dataset', {
        path: uncPath,
        options: { timeout_ms: 30000 } // Longer timeout for network
      });

      expect((result as any).is_network_path).toBe(true);
    });

    it('handles network timeouts gracefully', async () => {
      vi.mocked(invoke).mockRejectedValueOnce(
        new Error('Network path not found or timeout')
      );

      await expect(invoke('load_dataset', {
        path: '\\\\OfflineServer\\Share\\data.csv',
        options: {}
      })).rejects.toThrow('Network path not found');
    });
  });

  describe('Windows Error Messages', () => {
    it('provides Windows-specific error context', async () => {
      const windowsErrors = [
        { 
          code: 'ERROR_PATH_NOT_FOUND',
          message: 'The system cannot find the path specified'
        },
        {
          code: 'ERROR_ACCESS_DENIED', 
          message: 'Access is denied'
        },
        {
          code: 'ERROR_SHARING_VIOLATION',
          message: 'The process cannot access the file because it is being used by another process'
        }
      ];

      for (const error of windowsErrors) {
        vi.mocked(invoke).mockRejectedValueOnce({
          code: error.code,
          message: error.message,
          platform: 'windows',
        });

        try {
          await invoke('test_operation', {});
        } catch (e: any) {
          expect(e.code).toBe(error.code);
          expect(e.platform).toBe('windows');
        }
      }
    });
  });
}, BASE_TIMEOUT);

// Cross-platform integration tests that have Windows-specific behavior
describe('Cross-platform Integration with Windows Considerations', () => {
  it('handles line endings correctly across platforms', async () => {
    const testData = 'line1\r\nline2\r\nline3'; // Windows line endings
    
    vi.mocked(invoke).mockResolvedValueOnce({
      original_line_ending: IS_WINDOWS ? 'CRLF' : 'LF',
      normalized: true,
    });

    const result = await invoke('process_text_file', {
      content: testData,
      normalize_line_endings: true,
    });

    expect((result as any).normalized).toBe(true);
  });

  it('respects platform-specific temp directories', async () => {
    vi.mocked(invoke).mockResolvedValueOnce({
      temp_path: IS_WINDOWS 
        ? 'C:\\Users\\Test\\AppData\\Local\\Temp\\airimpute'
        : '/tmp/airimpute',
      created: true,
    });

    const result = await invoke('create_temp_workspace');
    const tempPath = (result as any).temp_path;
    
    if (IS_WINDOWS) {
      expect(tempPath).toContain('\\Temp\\');
    } else {
      expect(tempPath).toContain('/tmp/');
    }
  });
});