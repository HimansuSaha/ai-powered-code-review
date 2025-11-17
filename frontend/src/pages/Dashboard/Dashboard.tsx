import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
} from '@mui/material';
import {
  Security,
  Code,
  BugReport,
  TrendingUp,
  Assessment,
  Timeline,
} from '@mui/icons-material';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { useQuery } from 'react-query';
import { dashboardApi } from '../../services/api';

// Sample data for charts
const scoreData = [
  { date: '2024-01-01', score: 78.5 },
  { date: '2024-01-02', score: 80.1 },
  { date: '2024-01-03', score: 82.3 },
  { date: '2024-01-04', score: 79.8 },
  { date: '2024-01-05', score: 85.2 },
  { date: '2024-01-06', score: 87.1 },
  { date: '2024-01-07', score: 88.9 },
];

const issuesData = [
  { date: '2024-01-01', critical: 2, high: 5, medium: 8, low: 12 },
  { date: '2024-01-02', critical: 1, high: 4, medium: 7, low: 10 },
  { date: '2024-01-03', critical: 1, high: 3, medium: 6, low: 8 },
  { date: '2024-01-04', critical: 0, high: 2, medium: 5, low: 7 },
  { date: '2024-01-05', critical: 0, high: 1, medium: 4, low: 6 },
];

const issueDistribution = [
  { name: 'Critical', value: 4, color: '#f44336' },
  { name: 'High', value: 15, color: '#ff9800' },
  { name: 'Medium', value: 30, color: '#ffeb3b' },
  { name: 'Low', value: 43, color: '#4caf50' },
];

const StatCard: React.FC<{
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
  subtitle?: string;
}> = ({ title, value, icon, color, subtitle }) => {
  return (
    <Card elevation={2}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography variant="h4" component="div" fontWeight="bold">
              {value}
            </Typography>
            <Typography variant="h6" color="text.secondary">
              {title}
            </Typography>
            {subtitle && (
              <Typography variant="body2" color="text.secondary">
                {subtitle}
              </Typography>
            )}
          </Box>
          <Avatar sx={{ bgcolor: color, width: 56, height: 56 }}>
            {icon}
          </Avatar>
        </Box>
      </CardContent>
    </Card>
  );
};

const Dashboard: React.FC = () => {
  // Fetch dashboard data
  const { data: stats, isLoading: statsLoading } = useQuery(
    'dashboardStats',
    dashboardApi.getStats,
    {
      refetchInterval: 30000, // Refetch every 30 seconds
    }
  );

  const { data: trends, isLoading: trendsLoading } = useQuery(
    'dashboardTrends',
    dashboardApi.getTrends
  );

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
        Dashboard
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" gutterBottom>
        Monitor your code quality and security metrics
      </Typography>

      <Grid container spacing={3}>
        {/* Statistics Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Analyses"
            value={stats?.data?.total_analyses || 156}
            icon={<Assessment />}
            color="#1976d2"
            subtitle="This month"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Security Issues"
            value={stats?.data?.security_issues || 23}
            icon={<Security />}
            color="#f44336"
            subtitle="Requires attention"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Quality Issues"
            value={stats?.data?.quality_issues || 45}
            icon={<BugReport />}
            color="#ff9800"
            subtitle="Code improvements"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Average Score"
            value={`${stats?.data?.average_score || 78.5}%`}
            icon={<TrendingUp />}
            color="#4caf50"
            subtitle="Overall quality"
          />
        </Grid>

        {/* Quality Score Trend */}
        <Grid item xs={12} md={8}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Box display="flex" alignItems="center" mb={2}>
              <Timeline sx={{ mr: 1, color: '#1976d2' }} />
              <Typography variant="h6" fontWeight="bold">
                Quality Score Trend
              </Typography>
            </Box>
            <Box height={300}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={scoreData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke="#1976d2"
                    strokeWidth={3}
                    dot={{ fill: '#1976d2', r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Issue Distribution */}
        <Grid item xs={12} md={4}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Box display="flex" alignItems="center" mb={2}>
              <BugReport sx={{ mr: 1, color: '#f44336' }} />
              <Typography variant="h6" fontWeight="bold">
                Issue Distribution
              </Typography>
            </Box>
            <Box height={300}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={issueDistribution}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {issueDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Issues Over Time */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Box display="flex" alignItems="center" mb={2}>
              <Code sx={{ mr: 1, color: '#ff9800' }} />
              <Typography variant="h6" fontWeight="bold">
                Issues Over Time
              </Typography>
            </Box>
            <Box height={300}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={issuesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Area
                    type="monotone"
                    dataKey="critical"
                    stackId="1"
                    stroke="#f44336"
                    fill="#f44336"
                    fillOpacity={0.8}
                  />
                  <Area
                    type="monotone"
                    dataKey="high"
                    stackId="1"
                    stroke="#ff9800"
                    fill="#ff9800"
                    fillOpacity={0.8}
                  />
                  <Area
                    type="monotone"
                    dataKey="medium"
                    stackId="1"
                    stroke="#ffeb3b"
                    fill="#ffeb3b"
                    fillOpacity={0.8}
                  />
                  <Area
                    type="monotone"
                    dataKey="low"
                    stackId="1"
                    stroke="#4caf50"
                    fill="#4caf50"
                    fillOpacity={0.8}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Recent Analyses */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom fontWeight="bold">
              Recent Analyses
            </Typography>
            <List>
              {(stats?.data?.recent_analyses || [
                {
                  id: "1",
                  file_path: "src/main.py",
                  score: 85.2,
                  issues: 3,
                  timestamp: "2024-01-01T10:00:00Z"
                },
                {
                  id: "2", 
                  file_path: "src/utils.py",
                  score: 92.1,
                  issues: 1,
                  timestamp: "2024-01-01T09:30:00Z"
                }
              ]).map((analysis: any) => (
                <ListItem key={analysis.id} divider>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: analysis.score > 80 ? '#4caf50' : '#ff9800' }}>
                      <Code />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={analysis.file_path}
                    secondary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Chip 
                          label={`Score: ${analysis.score}%`} 
                          size="small" 
                          color={analysis.score > 80 ? 'success' : 'warning'}
                        />
                        <Chip 
                          label={`${analysis.issues} issues`} 
                          size="small" 
                          variant="outlined"
                        />
                      </Box>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom fontWeight="bold">
              Quick Actions
            </Typography>
            <Box display="flex" flexDirection="column" gap={2}>
              <Card variant="outlined" sx={{ cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}>
                <CardContent sx={{ py: 2 }}>
                  <Box display="flex" alignItems="center">
                    <Code sx={{ mr: 2, color: '#1976d2' }} />
                    <Typography variant="body1">
                      Analyze New Code
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
              
              <Card variant="outlined" sx={{ cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}>
                <CardContent sx={{ py: 2 }}>
                  <Box display="flex" alignItems="center">
                    <Security sx={{ mr: 2, color: '#f44336' }} />
                    <Typography variant="body1">
                      Security Audit
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
              
              <Card variant="outlined" sx={{ cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}>
                <CardContent sx={{ py: 2 }}>
                  <Box display="flex" alignItems="center">
                    <Assessment sx={{ mr: 2, color: '#4caf50' }} />
                    <Typography variant="body1">
                      View Reports
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;