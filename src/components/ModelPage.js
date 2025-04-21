import React from "react";
import { Container, Typography, AppBar, Toolbar, IconButton, ThemeProvider, createTheme, responsiveFontSizes } from "@mui/material";
import { Link } from "react-router-dom";
import HomeIcon from "@mui/icons-material/Home";
import ClipModelViz from "./ClipModelViz";

// Create a responsive theme with custom typography
let theme = createTheme({
  typography: {
    fontFamily: [
      'Roboto',
      '"Segoe UI"',
      'Arial',
      'sans-serif'
    ].join(','),
    h3: {
      fontWeight: 500,
      letterSpacing: '-0.5px',
    },
    h5: {
      fontWeight: 500,
      letterSpacing: '0.5px',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    }
  },
  palette: {
    primary: {
      main: "#003366", // Dark blue
    },
    secondary: {
      main: "#4a90e2", // Lighter blue for accents
    },
    background: {
      default: "#f8f9fa",
      paper: "#ffffff",
    },
  },
});

// Make typography responsive
theme = responsiveFontSizes(theme);

// Styling for dark blue background
const darkBlue = "#003366"; 

function ModelPage() {
  return (
    <ThemeProvider theme={theme}>
      <div style={{ backgroundColor: theme.palette.background.default, minHeight: '100vh' }}>
        {/* Menu Bar at the top with a dark blue background */}
        <AppBar position="sticky" sx={{ 
          backgroundColor: darkBlue,
          boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)'
        }}>
          <Toolbar>
            {/* Home Icon on the left */}
            <IconButton 
              edge="start" 
              color="inherit" 
              component={Link} 
              to="/" 
              aria-label="home"
              sx={{ 
                mr: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)'
                }
              }}
            >
              <HomeIcon />
            </IconButton>
            <Typography 
              variant="h6" 
              sx={{ 
                flexGrow: 1,
                fontWeight: 500,
                letterSpacing: '0.5px'
              }}
            >
              CLIP Explorer
            </Typography>
          </Toolbar>
        </AppBar>

        {/* Content of the ModelPage */}
        <Container sx={{ mt: 4, mb: 4, pt: 2 }}>
          <Typography 
            variant="h3" 
            gutterBottom
            sx={{
              fontWeight: 600,
              color: theme.palette.primary.main,
              mb: 3,
              fontSize: '1.8rem !important',
              position: 'relative',
              // '&:after': {
              //   content: '""',
              //   position: 'absolute',
              //   bottom: '-8px',
              //   left: 0,
              //   width: '60px',
              //   height: '4px',
              //   backgroundColor: theme.palette.secondary.main,
              //   borderRadius: '2px'
              // }
            }}
          >
            Exploring OpenAI's CLIP: A Visual Journey
          
          
          </Typography>
          
          <ClipModelViz />
        </Container>
      </div>
    </ThemeProvider>
  );
}

export default ModelPage;
