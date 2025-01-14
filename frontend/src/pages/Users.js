import React, { useState } from 'react';
import { Table, Card, Button, message, Modal, Form, Input, Space, Popconfirm } from 'antd';
import { PlusOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { userApi } from '../api';

const UserForm = ({ onSubmit, initialValues, loading }) => {
  const [form] = Form.useForm();

  return (
    <Form
      form={form}
      layout="vertical"
      onFinish={onSubmit}
      initialValues={initialValues || {}}
    >
      <Form.Item
        name="name"
        label="Name"
        rules={[{ required: true, message: 'Please input the name!' }]}
      >
        <Input placeholder="Enter full name" />
      </Form.Item>
      
      <Form.Item
        name="email"
        label="Email"
        rules={[
          { required: true, message: 'Please input the email!' },
          { type: 'email', message: 'Please enter a valid email!' }
        ]}
      >
        <Input placeholder="Enter email address" />
      </Form.Item>

      <Form.Item className="mb-0">
        <Button type="primary" htmlType="submit" loading={loading} block>
          {initialValues ? 'Update User' : 'Create User'}
        </Button>
      </Form.Item>
    </Form>
  );
};

const Users = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);
  const queryClient = useQueryClient();

  const { data: users, isLoading } = useQuery('users', userApi.getUsers);

  const updateMutation = useMutation(
    ({ id, ...data }) => userApi.updateUser(id, data),
    {
      onSuccess: () => {
        message.success('User updated successfully');
        setIsModalOpen(false);
        setSelectedUser(null);
        queryClient.invalidateQueries('users');
      },
      onError: (error) => {
        message.error(error.response?.data?.message || 'Failed to update user');
      }
    }
  );

  const createMutation = useMutation(userApi.createUser, {
    onSuccess: () => {
      message.success('User created successfully');
      setIsModalOpen(false);
      queryClient.invalidateQueries('users');
    },
    onError: (error) => {
      message.error(error.response?.data?.message || 'Failed to create user');
    }
  });

  const deleteMutation = useMutation(
    (id) => userApi.deleteUser(id),
    {
      onSuccess: () => {
        message.success('User deleted successfully');
        queryClient.invalidateQueries('users');
      },
      onError: (error) => {
        message.error(error.response?.data?.message || 'Failed to delete user');
      }
    }
  );

  const handleSubmit = (values) => {
    if (selectedUser?.id) {
      updateMutation.mutate({ id: selectedUser.id, ...values });
    } else {
      createMutation.mutate(values);
    }
  };

  const handleDeleteUser = (id) => {
    deleteMutation.mutate(id);
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => a.name.localeCompare(b.name),
    },
    {
      title: 'Email',
      dataIndex: 'email',
      key: 'email',
    },
    {
      title: 'Created At',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => new Date(date).toLocaleString(),
      sorter: (a, b) => new Date(a.created_at) - new Date(b.created_at),
    },
    {
      title: 'Training Sessions',
      dataIndex: 'training_sessions_count',
      key: 'training_sessions_count',
      render: (count) => count || 0,
      sorter: (a, b) => (a.training_sessions_count || 0) - (b.training_sessions_count || 0),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button 
            icon={<EditOutlined />}
            onClick={() => {
              setSelectedUser(record);
              setIsModalOpen(true);
            }}
          />
          <Popconfirm
            title="Delete User"
            description="Are you sure you want to delete this user?"
            onConfirm={() => handleDeleteUser(record.id)}
            okText="Yes"
            cancelText="No"
          >
            <Button icon={<DeleteOutlined />} danger />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div className="p-4">
      <Card 
        title="Users"
        extra={
          <Button 
            type="primary" 
            icon={<PlusOutlined />}
            onClick={() => {
              setSelectedUser(null);
              setIsModalOpen(true);
            }}
          >
            Add User
          </Button>
        }
      >
        <Table
          dataSource={users?.data}
          columns={columns}
          rowKey="id"
          loading={isLoading}
          scroll={{ x: true }}
        />
      </Card>

      <Modal
        title={selectedUser ? 'Edit User' : 'Create New User'}
        open={isModalOpen}
        onCancel={() => {
          setIsModalOpen(false);
          setSelectedUser(null);
        }}
        footer={null}
      >
        <UserForm 
          onSubmit={handleSubmit}
          initialValues={selectedUser}
          loading={createMutation.isLoading || updateMutation.isLoading}
        />
      </Modal>
    </div>
  );
};

export default Users;