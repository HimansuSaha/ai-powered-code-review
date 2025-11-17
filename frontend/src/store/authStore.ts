import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authApi } from '../services/api';
import toast from 'react-hot-toast';

interface User {
  id: string;
  email: string;
  full_name: string;
  organization?: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (userData: RegisterData) => Promise<void>;
  logout: () => void;
  setUser: (user: User) => void;
  setToken: (token: string) => void;
}

interface RegisterData {
  email: string;
  password: string;
  full_name: string;
  organization?: string;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true });
        try {
          const response = await authApi.login(email, password);
          const { access_token, user } = response.data;
          
          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
          });
          
          // Set token in API service
          authApi.setAuthToken(access_token);
          
          toast.success('Login successful!');
        } catch (error: any) {
          set({ isLoading: false });
          const message = error.response?.data?.detail || 'Login failed';
          toast.error(message);
          throw error;
        }
      },

      register: async (userData: RegisterData) => {
        set({ isLoading: true });
        try {
          const response = await authApi.register(userData);
          const { access_token, user } = response.data;
          
          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
          });
          
          // Set token in API service
          authApi.setAuthToken(access_token);
          
          toast.success('Registration successful!');
        } catch (error: any) {
          set({ isLoading: false });
          const message = error.response?.data?.detail || 'Registration failed';
          toast.error(message);
          throw error;
        }
      },

      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        });
        
        // Remove token from API service
        authApi.setAuthToken(null);
        
        toast.success('Logged out successfully');
      },

      setUser: (user: User) => {
        set({ user });
      },

      setToken: (token: string) => {
        set({ token, isAuthenticated: true });
        authApi.setAuthToken(token);
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);