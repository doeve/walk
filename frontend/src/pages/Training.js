import React, { useState } from 'react';
import { Card, Select, Row, Col, Upload, Button, message, Statistic, Progress, Timeline, Space, Switch } from 'antd';
import { 
  UploadOutlined, 
  VideoCameraOutlined, 
  UserOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  BarChartOutlined,
  InboxOutlined
} from '@ant-design/icons';
import Webcam from 'react-webcam';
import { useQuery } from 'react-query';
import { userApi, trainingApi } from '../api';

const { Dragger } = Upload;

const Training = () => {
  const [selectedUser, setSelectedUser] = useState(null);
  const [isLive, setIsLive] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const webcamRef = React.useRef(null);
  const [recordedChunks, setRecordedChunks] = useState([]);
  
  const { data: users } = useQuery('users', userApi.getUsers);
  const { data: trainingSessions } = useQuery(
    ['trainingSessions', selectedUser],
    () => selectedUser ? trainingApi.getTrainingSessions(selectedUser) : null,
    { enabled: !!selectedUser }
  );

  const handleTraining = async (file) => {
    if (!selectedUser) {
      message.error('Please select a user first');
      return;
    }

    const formData = new FormData();
    formData.append('video', file);
    formData.append('userId', parseInt(selectedUser));

    try {
      await trainingApi.uploadVideo(formData);
      message.success('Training video uploaded successfully');
    } catch (error) {
      message.error('Failed to upload training video');
    }
  };

  const startRecording = () => {
    setIsRecording(true);
    setRecordingDuration(0);
    setRecordedChunks([]);
    
    const interval = setInterval(() => {
      setRecordingDuration(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  };

  const stopRecording = () => {
    setIsRecording(false);
    // Stop recording and save video
    message.success('Recording completed');
  };

  //test stats
  const stats = {
    totalSessions: 24,
    successRate: 92,
    averageDuration: '2.5 min',
    lastSession: '2 hours ago'
  };

  return (
    <div className="p-4">
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card className="mb-4">
            <Space size="large">
              <Select
                style={{ width: 200 }}
                placeholder="Select User"
                onChange={setSelectedUser}
                className="mr-4"
              >
                {users?.data?.map(user => (
                  <Select.Option key={user.id} value={user.id}>
                    {user.name}
                  </Select.Option>
                ))}
              </Select>
              {selectedUser && (
                <span className="text-gray-600">
                  Selected user has {stats.totalSessions} training sessions
                </span>
              )}
            </Space>
          </Card>
        </Col>

        <Col span={16}>
          <Card 
            title={
              <div className="flex justify-between items-center w-full">
                <span className="flex items-center">
                  <VideoCameraOutlined className="mr-2" />
                  Training Session
                </span>
                <Switch
                  checked={isLive}
                  onChange={setIsLive}
                  checkedChildren="Live"
                  unCheckedChildren="Upload"
                />
              </div>
            }
            className="mb-4"
          >
            <div className="min-h-[400px] flex items-center justify-center">
              {isLive ? (
                <div className="w-full">
                  <div className="relative mb-4">
                    <Webcam
                      audio={false}
                      ref={webcamRef}
                      className="w-full rounded-lg"
                    />
                    {isRecording && (
                      <div className="absolute top-4 right-4 bg-red-500 text-white px-3 py-1 rounded-full animate-pulse">
                        Recording: {recordingDuration}s
                      </div>
                    )}
                  </div>
                  <div className="flex justify-center">
                    <Button
                      type={isRecording ? 'danger' : 'primary'}
                      icon={<VideoCameraOutlined />}
                      onClick={isRecording ? stopRecording : startRecording}
                      size="large"
                    >
                      {isRecording ? 'Stop Recording' : 'Start Recording'}
                    </Button>
                  </div>
                </div>
              ) : (
                <Dragger
                  accept="video/*"
                  beforeUpload={(file) => {
                    handleTraining(file);
                    return false;
                  }}
                  className="w-full"
                >
                  <p className="text-4xl mb-4">
                    <InboxOutlined />
                  </p>
                  <p className="text-lg mb-2">Click or drag video file to this area</p>
                  <p className="text-sm text-gray-500">Support for MP4, AVI, MOV formats</p>
                </Dragger>
              )}
            </div>
          </Card>
        </Col>

        <Col span={8}>
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Card title="Training Statistics">
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Statistic
                      title="Total Sessions"
                      value={stats.totalSessions}
                      prefix={<BarChartOutlined />}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="Success Rate"
                      value={stats.successRate}
                      suffix="%"
                      prefix={<CheckCircleOutlined />}
                    />
                  </Col>
                  <Col span={24} className="mt-4">
                    <Progress
                      percent={stats.successRate}
                      status="active"
                      strokeColor={{
                        '0%': '#108ee9',
                        '100%': '#87d068',
                      }}
                    />
                  </Col>
                </Row>
              </Card>
            </Col>

            <Col span={24}>
              <Card title="Recent Sessions">
                <Timeline
                  items={[
                    {
                      color: 'green',
                      children: 'Training completed successfully',
                    },
                    {
                      color: 'green',
                      children: 'Video processed and analyzed',
                    },
                    {
                      color: 'blue',
                      children: 'New gait pattern detected',
                    },
                    {
                      color: 'gray',
                      children: 'Session started',
                    },
                  ]}
                />
              </Card>
            </Col>
          </Row>
        </Col>
      </Row>
    </div>
  );
};

export default Training;