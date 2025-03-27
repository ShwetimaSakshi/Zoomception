import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import { Typography, Button, Box, Paper } from "@mui/material";
import { styled } from "@mui/system";
import videoBackground from "./assets/video.mp4"; // Import the video
import ModelPage from "./components/ModelPage";
import AttentionPage from "./components/AttentionPage";

const Background = styled(Box)({
  height: "100vh",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  position: "relative", // Important to overlay content on top
  overflow: "hidden",
});

const VideoBackground = styled("video")({
  position: "absolute", // Position it behind the content
  top: 0,
  left: 0,
  width: "100%", // Stretch to full screen
  height: "100%",
  objectFit: "cover", // Make sure video covers the whole screen
  zIndex: -1, // Place the video behind the content
});

const Banner = styled(Paper)({
  padding: "40px",
  textAlign: "center",
  background: "rgba(0, 0, 0, 0.7)", // Semi-transparent black background for the banner
  borderRadius: "15px",
  maxWidth: "600px",
  color: "white",
});

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AboutPage />} />
        <Route path="/attention" element={<AttentionPage />} />
        <Route path="/model" element={<ModelPage />} />
      </Routes>
    </Router>
  );
}

function AboutPage() {
  return (
    <Background>
      <VideoBackground autoPlay loop muted>
        <source src={videoBackground} type="video/mp4" />
      </VideoBackground>

      <Banner elevation={10}>
        <Typography variant="h3" gutterBottom>
          About Us
        </Typography>
        <Typography variant="body1" paragraph>
          Welcome to our data visualization platform. Explore models and attention mapping with interactive insights.
        </Typography>
        <Box mt={4} display="flex" justifyContent="center" gap={2}>
          <Button variant="contained" color="primary" component={Link} to="/model">
            Explore Model
          </Button>
          <Button variant="contained" color="secondary" component={Link} to="/attention">
            Explore Attention
          </Button>
        </Box>
      </Banner>
    </Background>
  );
}

export default App;
