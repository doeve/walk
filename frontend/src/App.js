import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from 'antd';
import AppNavigation from './components/AppNavigation';
import GaitRecognition from './pages/GaitRecognition';
import Training from './pages/Training';
import Users from './pages/Users';

const { Sider, Content } = Layout;

function App() {
  return (
    <Router>
      <Layout className="min-h-screen">
        <Sider theme="light" width={200} className="border-r border-gray-200">
          <AppNavigation />
        </Sider>
        <Content className="p-6 bg-white">
          <Routes>
            <Route path="/" element={<GaitRecognition />} />
            <Route path="/training" element={<Training />} />
            <Route path="/users" element={<Users />} />
          </Routes>
        </Content>
      </Layout>
    </Router>
  );
}

export default App;