import React, { useState } from 'react';
import { Row, Col, Card, List, Progress, Upload, Switch } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import Webcam from 'react-webcam';

const { Dragger } = Upload;

const SubjectGuessingPage = () => {
  const [isLive, setIsLive] = useState(false);
  const [predictions, setPredictions] = useState([]);

  const handlePrediction = (file) => {
    // Handle prediction logic here
    console.log('Processing file:', file);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <Row gutter={[24, 24]} className="h-full">
        <Col xs={24} xl={12}>
          <Card 
            title={
              <div className="flex justify-between items-center w-full">
                <span>Video Input</span>
                <Switch
                  checked={isLive}
                  onChange={setIsLive}
                  checkedChildren="Live"
                  unCheckedChildren="Upload"
                />
              </div>
            }
            className="h-full min-h-[500px] shadow-md"
            styles={{
              body: { 
                height: 'calc(100% - 58px)',
                overflowY: 'auto'
              }
            }}
          >
            <div className="flex items-center justify-center h-full">
              {isLive ? (
                <div className="w-full h-full flex items-center justify-center">
                  <Webcam
                    audio={false}
                    className="w-full max-h-[400px] rounded-lg object-cover"
                  />
                </div>
              ) : (
                <Dragger
                  accept="video/*"
                  beforeUpload={(file) => {
                    handlePrediction(file);
                    return false;
                  }}
                  className="w-full h-full flex items-center justify-center border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-500 transition-colors"
                >
                  <div className="text-center p-8">
                    <p className="text-4xl mb-4 text-gray-400">
                      <InboxOutlined />
                    </p>
                    <p className="text-lg mb-2">
                      Click or drag video file to this area
                    </p>
                    <p className="text-sm text-gray-500">
                      Support for video files only
                    </p>
                  </div>
                </Dragger>
              )}
            </div>
          </Card>
        </Col>
        
        <Col xs={24} xl={12}>
          <Card 
            title="Predictions" 
            className="h-full min-h-[500px] shadow-md"
            styles={{
              body: { 
                height: 'calc(100% - 58px)',
                overflowY: 'auto'
              }
            }}
          >
            {predictions.length > 0 ? (
              <List
                dataSource={predictions}
                className="prediction-list"
                renderItem={(prediction) => (
                  <List.Item className="py-4">
                    <List.Item.Meta
                      title={
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-semibold">{prediction.name}</span>
                          <span className="text-sm text-gray-500">
                            {(prediction.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      }
                      description={
                        <Progress 
                          percent={prediction.confidence * 100} 
                          status="active"
                          strokeColor={{
                            '0%': '#108ee9',
                            '100%': '#87d068',
                          }}
                        />
                      }
                    />
                  </List.Item>
                )}
              />
            ) : (
              <div className="h-full flex items-center justify-center text-gray-400">
                <div className="text-center">
                  <p className="text-6xl mb-4">📊</p>
                  <p>No predictions yet</p>
                  <p className="text-sm">Upload or record a video to start</p>
                </div>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default SubjectGuessingPage;