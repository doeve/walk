import React from 'react';
import { Menu } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  VideoCameraOutlined, 
  UserOutlined, 
  ExperimentOutlined 
} from '@ant-design/icons';

const AppNavigation = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const items = [
    {
      key: '/',
      icon: <VideoCameraOutlined />,
      label: 'Gait Recognition'
    },
    {
      key: '/training',
      icon: <ExperimentOutlined />,
      label: 'Training'
    },
    {
      key: '/users',
      icon: <UserOutlined />,
      label: 'Users'
    }
  ];

  return (
    <div className="h-full">
      <Menu
        mode="inline"
        selectedKeys={[location.pathname]}
        items={items}
        onClick={({ key }) => navigate(key)}
        className="border-none"
      />
    </div>
  );
};

export default AppNavigation;