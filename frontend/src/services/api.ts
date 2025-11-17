import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Auth API
export const authApi = {
  setAuthToken: (token: string | null) => {
    if (token) {
      apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      delete apiClient.defaults.headers.common['Authorization'];
    }
  },

  login: (email: string, password: string) => {
    return apiClient.post('/auth/login', { email, password });
  },

  register: (userData: {
    email: string;
    password: string;
    full_name: string;
    organization?: string;
  }) => {
    return apiClient.post('/auth/register', userData);
  },

  refreshToken: (refreshToken: string) => {
    return apiClient.post('/auth/refresh', {}, {
      headers: { Authorization: `Bearer ${refreshToken}` }
    });
  },

  getCurrentUser: () => {
    return apiClient.get('/auth/me');
  },

  logout: () => {
    return apiClient.post('/auth/logout');
  },
};

// Analysis API
export const analysisApi = {
  analyzeCode: (data: {
    code_content: string;
    file_path: string;
    language: string;
    analysis_type?: string;
  }) => {
    return apiClient.post('/analysis/analyze', data);
  },

  batchAnalyze: (data: {
    files: Array<{
      path: string;
      content: string;
      language: string;
    }>;
    repository_id?: string;
    commit_sha?: string;
    analysis_type?: string;
  }) => {
    return apiClient.post('/analysis/batch-analyze', data);
  },

  getAnalysisResults: (analysisId: string) => {
    return apiClient.get(`/analysis/results/${analysisId}`);
  },

  getSupportedLanguages: () => {
    return apiClient.get('/analysis/supported-languages');
  },

  getModelsStatus: () => {
    return apiClient.get('/analysis/models/status');
  },
};

// Dashboard API
export const dashboardApi = {
  getStats: () => {
    return apiClient.get('/dashboard/stats');
  },

  getTrends: () => {
    return apiClient.get('/dashboard/trends');
  },
};

// Repository API
export const repositoryApi = {
  getRepositories: () => {
    return apiClient.get('/repositories');
  },

  createRepository: (data: any) => {
    return apiClient.post('/repositories', data);
  },

  getRepository: (repositoryId: string) => {
    return apiClient.get(`/repositories/${repositoryId}`);
  },

  updateRepository: (repositoryId: string, data: any) => {
    return apiClient.put(`/repositories/${repositoryId}`, data);
  },

  deleteRepository: (repositoryId: string) => {
    return apiClient.delete(`/repositories/${repositoryId}`);
  },
};

// Add request interceptor to handle auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth-storage');
    if (token) {
      try {
        const parsedAuth = JSON.parse(token);
        if (parsedAuth.state?.token) {
          config.headers.Authorization = `Bearer ${parsedAuth.state.token}`;
        }
      } catch (error) {
        console.error('Error parsing auth token:', error);
      }
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle auth errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('auth-storage');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default apiClient;