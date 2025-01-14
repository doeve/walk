import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const userApi = {
  getUsers: () => api.get('/users/'),
  getUser: (id) => api.get(`/users/${id}`),
  createUser: (userData) => api.post('/users', userData),
  updateUser: (id, userData) => api.put(`/users/${id}`, userData),
  deleteUser: (id) => api.delete(`/users/${id}`),
};

export const trainingApi = {
  uploadVideo: (formData) => api.post('/training/upload/', formData),
  getTrainingSessions: (userId) => api.get(`/training/${userId}`),
};

export const predictionApi = {
  predict: (formData) => api.post('/predict', formData),
};

export default api;