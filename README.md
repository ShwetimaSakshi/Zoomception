# Zoomception

## Our app can be viewed at:  
👉 [https://zoomception.netlify.app/](https://zoomception.netlify.app/)

---

# Command-Line Deployment Steps for React App on Netlify

## Step 1: Install Netlify CLI

```bash
npm install -g netlify-cli
```

## Step 2: Initialize Project with Netlify

Navigate to the project directory and run:

```bash
netlify init
```

Follow the prompts to:
- Connect your GitHub repository  
- Choose the site name  
- Set the build and publish directories (e.g., `npm run build` for the build command and `build` for the publish directory)

## Step 3: Build Your Project

Build your project for production:

```bash
npm run build
```

This will create a `build` folder containing the optimized production version of your app.

## Step 4: Deploy Your Site

Deploy your app to Netlify using:

```bash
netlify deploy --prod
```

Netlify will:
1. Upload your build directory to the server.
2. Provide you with a live URL to preview your app.

(*Using `--prod` will perform a production deployment.*)

## Step 5: Check the Deployment URL

That's it — your React app is now deployed via the command line using Netlify!

---

# Steps to Run a React App Locally from GitHub

## Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/ShwetimaSakshi/Zoomception.git
```

## Step 2: Navigate to the Project Directory

```bash
cd zoomception
```

## Step 3: Install Dependencies

Install all required dependencies:

```bash
npm install
```

This will install the packages listed in the `package.json` file.

## Step 4: Start the Development Server

Start the React development server:

```bash
npm start
```

You should see an output like:

```
Compiled successfully!
```

Your app will now be available at:

- Local: [http://localhost:3000](http://localhost:3000)  
- On Your Network: http://192.168.x.x:3000  

## Step 5: Open the App in a Browser

Go to:

[http://localhost:3000](http://localhost:3000)

You should see your React app running locally.

---
