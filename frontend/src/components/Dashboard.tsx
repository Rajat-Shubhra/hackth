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

interface DashboardStats {
  total_transactions: number;
  fraud_transactions: number;
  model_status: string;
  fraud_rate?: number;
}

interface Transaction {
  transaction_id: number;
  amount: number;
  merchant: string;
  is_fraud: boolean;
  fraud_score: number;
  risk_level: string;
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<Transaction | null>(null);

  // Updated API URL to match our working backend
  const API_BASE = 'http://localhost:8000';

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Fetch dashboard stats
      const statsResponse = await fetch(`${API_BASE}/api/stats`);
      const statsData = await statsResponse.json();
      
      setStats(statsData);
      setLoading(false);
    } catch (err) {
      setError(`Failed to fetch dashboard data: ${err}`);
      setLoading(false);
    }
  };

  const testFraudDetection = async (amount: number, merchant: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          amount: amount,
          merchant: merchant,
          card_type: 'credit'
        })
      });
      
      const result = await response.json();
      setTestResult(result);
    } catch (err) {
      setError(`Failed to test fraud detection: ${err}`);
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
        <Button onClick={() => window.location.reload()} sx={{ mt: 2 }}>
          Retry
        </Button>
      </Container>
    );
  }

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
          onClick={() => testFraudDetection(15000, "Luxury Store")}
          sx={{ mr: 2 }}
          color="error"
        >
          Test High-Risk Transaction ($15,000)
        </Button>
        <Button 
          variant="contained"
          onClick={() => testFraudDetection(50, "Coffee Shop")}
          sx={{ mr: 2 }}
          color="success"
        >
          Test Normal Transaction ($50)
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
                <Security color={stats?.model_status === 'active' ? 'success' : 'warning'} sx={{ mr: 2, fontSize: 40 }} />
                <Box>
                  <Typography variant="h6">{stats?.model_status === 'active' ? 'ACTIVE' : 'FALLBACK'}</Typography>
                  <Typography color="text.secondary">ML Model Status</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <CheckCircle color="success" sx={{ mr: 2, fontSize: 40 }} />
                <Box>
                  <Typography variant="h4">{stats?.fraud_rate || 0}%</Typography>
                  <Typography color="text.secondary">Fraud Rate</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Test Results */}
      {testResult && (
        <Grid container spacing={3} mb={4}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                Latest Test Result
              </Typography>
              <Card>
                <CardContent>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={12} sm={3}>
                      <Typography variant="h6">
                        ${testResult.amount}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {testResult.merchant}
                      </Typography>
                    </Grid>
                    
                    <Grid item xs={12} sm={3}>
                      <Chip 
                        label={testResult.is_fraud ? 'FRAUD DETECTED' : 'LEGITIMATE'}
                        color={testResult.is_fraud ? 'error' : 'success'}
                        variant="filled"
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={3}>
                      <Typography variant="body1">
                        Risk: {testResult.risk_level}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Score: {(testResult.fraud_score * 100).toFixed(1)}%
                      </Typography>
                    </Grid>
                    
                    <Grid item xs={12} sm={3}>
                      <Typography variant="body2" color="text.secondary">
                        Transaction ID: {testResult.transaction_id}
                      </Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Paper>
          </Grid>
        </Grid>
      )}

      {/* API Status */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              System Status
            </Typography>
            <Alert severity="success">
              ✅ Backend API: Connected to {API_BASE}
            </Alert>
            <Alert severity="info" sx={{ mt: 1 }}>
              🤖 ML Model: {stats?.model_status === 'active' ? 'Production RandomForest model loaded' : 'Using fallback rules'}
            </Alert>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
