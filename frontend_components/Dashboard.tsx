import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Button,
  Chip
} from '@mui/material';
import {
  TrendingUp,
  Security,
  Warning,
  CheckCircle
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface DashboardStats {
  total_transactions: number;
  fraud_transactions: number;
  pending_alerts: number;
  fraud_rate: number;
}

interface Transaction {
  id: number;
  amount: number;
  user_id: string;
  merchant_id: string;
  is_fraud: boolean;
  fraud_score: number;
  timestamp: string;
  transaction_type: string;
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentTransactions, setRecentTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = 'http://localhost:5000';

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Fetch dashboard stats
      const statsResponse = await fetch(`${API_BASE}/api/dashboard/stats`);
      const statsData = await statsResponse.json();
      
      // Fetch recent transactions
      const transactionsResponse = await fetch(`${API_BASE}/api/transactions/?per_page=10`);
      const transactionsData = await transactionsResponse.json();
      
      setStats(statsData);
      setRecentTransactions(transactionsData.transactions || []);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch dashboard data');
      setLoading(false);
    }
  };

  const generateSampleData = async () => {
    try {
      await fetch(`${API_BASE}/api/transactions/sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 20 })
      });
      fetchDashboardData(); // Refresh data
    } catch (err) {
      setError('Failed to generate sample data');
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress size={60} />
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  const fraudRateColor = stats && stats.fraud_rate > 10 ? 'error' : stats && stats.fraud_rate > 5 ? 'warning' : 'success';

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box mb={4}>
        <Typography variant="h3" component="h1" gutterBottom>
          🛡️ Fraud Detection Dashboard
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Real-time AI-powered fraud monitoring for financial transactions
        </Typography>
      </Box>

      {/* Action Buttons */}
      <Box mb={4}>
        <Button 
          variant="contained" 
          onClick={generateSampleData}
          sx={{ mr: 2 }}
        >
          Generate Sample Data
        </Button>
        <Button 
          variant="outlined"
          onClick={fetchDashboardData}
        >
          Refresh Data
        </Button>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <TrendingUp color="primary" sx={{ mr: 2, fontSize: 40 }} />
                <Box>
                  <Typography variant="h4">{stats?.total_transactions || 0}</Typography>
                  <Typography color="text.secondary">Total Transactions</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Warning color="error" sx={{ mr: 2, fontSize: 40 }} />
                <Box>
                  <Typography variant="h4">{stats?.fraud_transactions || 0}</Typography>
                  <Typography color="text.secondary">Fraud Detected</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Security color="warning" sx={{ mr: 2, fontSize: 40 }} />
                <Box>
                  <Typography variant="h4">{stats?.pending_alerts || 0}</Typography>
                  <Typography color="text.secondary">Pending Alerts</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <CheckCircle color={fraudRateColor} sx={{ mr: 2, fontSize: 40 }} />
                <Box>
                  <Typography variant="h4">{stats?.fraud_rate || 0}%</Typography>
                  <Typography color="text.secondary">Fraud Rate</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Transactions */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Recent Transactions
            </Typography>
            
            {recentTransactions.length === 0 ? (
              <Alert severity="info">
                No transactions found. Generate some sample data to get started!
              </Alert>
            ) : (
              <Box>
                {recentTransactions.map((transaction) => (
                  <Card key={transaction.id} sx={{ mb: 2 }}>
                    <CardContent>
                      <Grid container spacing={2} alignItems="center">
                        <Grid item xs={12} sm={3}>
                          <Typography variant="h6">
                            ${transaction.amount}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {transaction.transaction_type}
                          </Typography>
                        </Grid>
                        
                        <Grid item xs={12} sm={3}>
                          <Typography variant="body1">
                            {transaction.user_id}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            User ID
                          </Typography>
                        </Grid>
                        
                        <Grid item xs={12} sm={3}>
                          <Chip 
                            label={transaction.is_fraud ? 'FRAUD' : 'LEGITIMATE'}
                            color={transaction.is_fraud ? 'error' : 'success'}
                            variant={transaction.is_fraud ? 'filled' : 'outlined'}
                          />
                          <Typography variant="body2" color="text.secondary">
                            Score: {(transaction.fraud_score * 100).toFixed(1)}%
                          </Typography>
                        </Grid>
                        
                        <Grid item xs={12} sm={3}>
                          <Typography variant="body2" color="text.secondary">
                            {new Date(transaction.timestamp).toLocaleString()}
                          </Typography>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                ))}
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
