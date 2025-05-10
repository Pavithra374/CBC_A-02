# EnginSync Platform Frontend

EnginSync is an educational platform designed for engineering students that provides various modules for managing educational content, tracking progress, planning activities, and personalized learning experiences.

## Project Structure

The project is built with React and Redux Toolkit, using a modular approach to organize code by feature. Here's an overview of the main directories:

- `src/features/`: Contains feature-specific components organized by module
  - `auth/`: Authentication components (Login, Register)
  - `planner/`: Planner module for managing events and tasks
  - `progress/`: Progress tracking and assessment components
  - `courses/`: Course management and listing components
  - `textbookbot/`: AI-powered assistant for answering questions about course materials
  - `placement/`: Placement assessment module for personalizing learning paths
  - `settings/`: User settings and preferences management
  - `dashboard/`: Main dashboard overview components

- `src/store/`: Redux state management
  - `index.js`: Root Redux store configuration
  - `slices/`: Redux Toolkit slices for state management
  - `api/`: RTK Query API services organized by feature

- `src/components/`: Shared UI components used across features
  - `layout/`: Layout components including navigation and common UI elements

## Getting Started

### Prerequisites

- Node.js (v14.0 or later)
- npm (v6.0 or later)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   npm install
   ```
3. Start the development server:
   ```
   npm start
   ```

The application will be available at `http://localhost:3000`.

## API Integration

The frontend is designed to work with a Flask backend following the API specifications defined in the `enginsync-api-docs` directory. The RTK Query services are already set up to connect with the corresponding backend endpoints.

## Features

### Authentication

- JWT-based authentication with token refresh mechanism
- Secure user login and registration
- Protected routes requiring authentication

### Planner Module

- Create, view, update, and delete tasks and events
- Calendar view for visualizing scheduled items
- Upcoming and today's events widgets

### Progress Tracking

- Course completion tracking
- Assessment results and analytics
- Learning goal management

### Adaptive Learning

- Personalized learning profiles
- Custom study plans based on performance
- Recommendations for learning resources

### User Interface

- Modern Material-UI components
- Responsive design for desktop and mobile
- Intuitive navigation and user experience

## Development

### State Management

The application uses Redux Toolkit for state management with RTK Query for API calls. Each feature has its own API slice for handling data fetching, caching, and updates.

### Authentication Flow

1. User logs in, receiving an access token and refresh token
2. Access token is stored in Redux state and used for API requests
3. If the access token expires, the refresh token is used to request a new one
4. If the refresh token is invalid, the user is logged out

### Adding New Features

1. Create a new directory in `src/features/` for the feature
2. Add the feature's components in this directory
3. Create an API slice in `src/store/api/` if needed
4. Update the Redux store to include the new slice
5. Add routes to `App.js`

## Deployment

To build the application for production:

```
npm run build
```

This will create an optimized production build in the `build` directory, which can be deployed to any static hosting service.

## Backend Integration

This frontend application is designed to be paired with a Flask backend that implements the API specifications defined in the `enginsync-api-docs` directory.

## License

[MIT License](LICENSE)
