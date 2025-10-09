const express = require("express");
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const cors = require("cors");
const bcrypt = require("bcryptjs");
const path = require("path");

const app = express();

// ===== Middleware =====
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(__dirname));

// ===== MongoDB Atlas Connection =====
// ADD DATABASE NAME AFTER THE CLUSTER URL
const MONGODB_URI = "mongodb+srv://jkgang432:johndoe123@cluster0.lvpe7rz.mongodb.net/test?retryWrites=true&w=majority&appName=Cluster0";

console.log("Connecting to MongoDB Atlas...");

mongoose.connect(MONGODB_URI, {
  serverSelectionTimeoutMS: 10000, // Increase timeout to 10 seconds
})
.then(() => {
  console.log("MongoDB Atlas Connected Successfully!");
  console.log("Database:", mongoose.connection.db.databaseName);
})

// ===== User Schema & Model =====
const UserSchema = new mongoose.Schema({
  username: { 
    type: String, 
    unique: true, 
    required: true,
    trim: true,
    minlength: 3,
    maxlength: 20
  },
  email: { 
    type: String, 
    unique: true, 
    required: true,
    trim: true,
    lowercase: true
  },
  password: { 
    type: String, 
    required: true,
    minlength: 6
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

const User = mongoose.model("User", UserSchema);

// ===== Routes =====

// Signup Endpoint
app.post("/signup", async (req, res) => {
  try {
    const { username, email, password } = req.body;

    // Validate fields
    if (!username || !email || !password) {
      return res.status(400).json({ error: "All fields are required" });
    }

    if (password.length < 6) {
      return res.status(400).json({ error: "Password must be at least 6 characters long" });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 12);

    // Create and save user
    const user = new User({ 
      username: username.trim(),
      email: email.toLowerCase().trim(),
      password: hashedPassword 
    });

    await user.save();

    console.log("New user registered:", username);
    res.json({ 
      success: true,
      message: "User registered successfully!" 
    });

  } catch (err) {
    console.error("Signup error:", err.message);
    
    // Handle duplicate key errors
    if (err.code === 11000) {
      const field = Object.keys(err.keyPattern)[0];
      return res.status(400).json({ 
        error: `${field} already exists` 
      });
    }
    
    res.status(500).json({ error: "Server error during registration" });
  }
});

// Login Endpoint
app.post("/login", async (req, res) => {
  try {
    const { identifier, password } = req.body;

    console.log("Login attempt for:", identifier);

    if (!identifier || !password) {
      return res.status(400).json({ 
        success: false, 
        message: "Please fill in all fields" 
      });
    }

    // Find user by email or username (case-insensitive)
    const user = await User.findOne({
      $or: [
        { email: identifier.toLowerCase().trim() },
        { username: identifier.trim() }
      ]
    });

    console.log("User found:", user ? "Yes" : "No");

    if (!user) {
      return res.status(400).json({ 
        success: false, 
        message: "User not found. Please sign up first." 
      });
    }

    // Compare passwords
    console.log("Comparing passwords...");
    const isMatch = await bcrypt.compare(password, user.password);
    console.log("Password match:", isMatch);
    
    if (!isMatch) {
      return res.status(400).json({ 
        success: false, 
        message: "Invalid password." 
      });
    }

    console.log("User logged in:", user.username);
    res.json({ 
      success: true, 
      message: "Login successful!",
      username: user.username 
    });

  } catch (error) {
    console.error("Login error details:", error);
    console.error("Error stack:", error.stack);
    res.status(500).json({ 
      success: false, 
      message: "Server error during login: " + error.message 
    });
  }
});

// Reset Password Endpoint
app.post("/reset-password", async (req, res) => {
  try {
    const { identifier, newPassword } = req.body;

    if (!identifier || !newPassword) {
      return res.status(400).json({ error: "All fields are required" });
    }

    if (newPassword.length < 6) {
      return res.status(400).json({ error: "Password must be at least 6 characters long" });
    }

    const user = await User.findOne({
      $or: [
        { email: identifier.toLowerCase().trim() },
        { username: identifier.trim() }
      ]
    });

    if (!user) {
      return res.status(400).json({ error: "User not found" });
    }

    // Hash new password
    const hashedPassword = await bcrypt.hash(newPassword, 12);
    user.password = hashedPassword;
    await user.save();

    console.log("Password reset for:", user.username);
    res.json({ message: "Password reset successful. Please log in." });

  } catch (err) {
    console.error("Reset password error:", err);
    res.status(500).json({ error: "Server error during password reset" });
  }
});

// Health check endpoint
app.get("/health", async (req, res) => {
  try {
    // Check if MongoDB is connected
    await mongoose.connection.db.admin().ping();
    res.json({ 
      status: "healthy", 
      database: "connected",
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    res.status(500).json({ 
      status: "unhealthy", 
      database: "disconnected",
      error: err.message 
    });
  }
});

// Test database connection
app.get("/test-db", async (req, res) => {
  try {
    const userCount = await User.countDocuments();
    res.json({ 
      message: "Database is working!",
      userCount: userCount,
      database: mongoose.connection.db.databaseName
    });
  } catch (err) {
    res.status(500).json({ error: "Database error: " + err.message });
  }
});

// Serve HTML files
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.get("/upload.html", (req, res) => {
  res.sendFile(path.join(__dirname, "upload.html"));
});

app.get("/prediction.html", (req, res) => {
  res.sendFile(path.join(__dirname, "prediction.html"));
});

// ===== Start Server =====
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Access your website at: http://localhost:${PORT}`);
  console.log(`Using MongoDB Atlas`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Test DB: http://localhost:${PORT}/test-db`);
});