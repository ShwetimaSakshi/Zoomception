import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import { Typography, Button, Box, Paper, Stack } from "@mui/material";
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
  background: "rgba(5, 19, 67, 0.68)", // Semi-transparent black background for the banner
  borderRadius: "100%",
  maxWidth: "600px",
  height: "420px",
  color: "white",
  justifyContent: "center",
  alignItems: "center",
  alignContent: "center"
});

const feedbackFormUrl = "https://docs.google.com/forms/d/e/1FAIpQLSf1vy277bfmsvuprtt4aGuJ-_8K2BzmJhewkH5SSy5RvS-22g/viewform?usp=header"; 

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
    <div>
      <Background>
        <VideoBackground autoPlay loop muted>
          <source src={videoBackground} type="video/mp4" />
        </VideoBackground>

        <Stack alignItems="center">
          <Banner elevation={10}>
            <Typography variant="h3" gutterBottom>
              Welcome!
            </Typography>
              
            <Typography variant="body1" sx={{maxWidth: "400px"}}>
              This platform is built to help you <strong>explore, understand, and interact</strong> with CLIP model.
              Help us make this system smarter, clearer, and more useful by sharing your feedback.
            </Typography>
            <Box mt={4} display="flex" justifyContent="center" gap={2}>
              <Button
                variant="contained"
                sx={{
                  color: "white",
                  padding: "12px 24px",
                  textTransform: "none",
                  fontSize: "16px",
                  fontWeight: "bold",
                  '&:hover': {
                    background: "rgb(42, 62, 213)",
                  }
                }}
                color="primary"
                component={Link}
                to="/model"
              >
                🧩 Explore Model
              </Button>
              <Button
                variant="contained"
                sx={{
                  color: "white",
                  padding: "12px 24px",
                  textTransform: "none",
                  fontSize: "16px",
                  fontWeight: "bold",
                  '&:hover': {
                    background: "rgb(133, 15, 165)",
                  }
                }}
                color="secondary"
                component={Link}
                to="/attention"
              >
                📊 Explore Attention
              </Button>
            </Box>

            <Box mt={4} display="flex" justifyContent="center">
              <Button
                variant="contained"
                sx={{
                  background: "linear-gradient(45deg, #7F56D9, #4E73DF)",
                  color: "white",
                  padding: "12px 24px",
                  textTransform: "none",
                  fontSize: "16px",
                  fontWeight: "bold",
                  '&:hover': {
                    background: "linear-gradient(45deg, #6A42A8, #3E63B6)",
                  }
                }}
                href={feedbackFormUrl}
                target="_blank"
              >
                ✍️ Feedback
              </Button>
            </Box>
          </Banner>
        </Stack>
      </Background>
    </div>
  );
}


export default App;
