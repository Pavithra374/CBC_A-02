// --- Constants ---
const API_BASE_URL = 'http://127.0.0.1:8000/api/v1/planner/events/';

// --- State ---
let currentEvents = []; // Store fetched events globally for widget processing

// --- DOM Elements Cache (Wait for DOMContentLoaded) ---
let taskListUl, upcomingListUl, todayListUl, addEventBtn, eventModal, modalCloseBtn, modalCancelBtn, eventForm, modalTitle, eventIdInput, eventTitleInput, eventTypeInput, eventCourseInput, eventDateInput, eventTimeInput, eventNotesInput;

// --- Helper Functions ---
function formatDate(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);

    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    // Compare date parts only
    today.setHours(0, 0, 0, 0);
    tomorrow.setHours(0, 0, 0, 0);
    yesterday.setHours(0, 0, 0, 0);
    const inputDateOnly = new Date(date.getFullYear(), date.getMonth(), date.getDate());

    if (inputDateOnly.getTime() === today.getTime()) return 'Today';
    if (inputDateOnly.getTime() === tomorrow.getTime()) return 'Tomorrow';
    if (inputDateOnly.getTime() === yesterday.getTime()) return 'Yesterday';

    // Default format: MMM DD
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatTime(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
}

// Format date/time from form inputs to ISO string UTC for API
function formatToISOUtc(dateStr, timeStr) {
    if (!dateStr) return null;
    const localDate = new Date(`${dateStr}${timeStr ? 'T' + timeStr : 'T00:00:00'}`);
    if (isNaN(localDate)) return null; // Invalid date/time input
    return localDate.toISOString();
}

// Format ISO string UTC to YYYY-MM-DD for date input
function formatIsoToDateInput(isoString) {
    if (!isoString) return '';
    try {
        const date = new Date(isoString);
        return date.toISOString().split('T')[0];
    } catch (e) {
        return '';
    }
}

// Format ISO string UTC to HH:mm for time input
function formatIsoToTimeInput(isoString) {
    if (!isoString) return '';
    try {
        const date = new Date(isoString);
        const hours = date.getHours().toString().padStart(2, '0');
        const minutes = date.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    } catch (e) {
        return '';
    }
}

// Placeholder for getting the auth token (replace with actual implementation)
function getAuthToken() {
    // Example: Retrieve from localStorage, sessionStorage, or cookie
    // return localStorage.getItem('authToken'); 
    // For now, return null or a dummy token if your backend allows testing without auth
    console.warn('Using dummy auth token placeholder!');
    return 'dummy-token-replace-me'; // Replace this!
}

// API Call Helper
async function apiCall(url, method = 'GET', body = null) {
    const token = getAuthToken();
    const headers = {
        'Content-Type': 'application/json',
    };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`; // Assuming JWT Bearer token
    }

    const config = {
        method: method,
        headers: headers,
    };

    if (body && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
        config.body = JSON.stringify(body);
    }

    try {
        const response = await fetch(url, config);

        if (!response.ok) {
            // Handle specific errors like 401 Unauthorized
            if (response.status === 401) {
                // Redirect to login or show an error message
                console.error('Authentication Error:', response.statusText);
                alert('Authentication failed. Please log in again.');
                // Example: window.location.href = '/login'; 
                throw new Error('Unauthorized');
            }
            // Handle other errors
            const errorData = await response.json().catch(() => ({ message: response.statusText }));
            console.error(`API Error (${response.status}):`, errorData);
            throw new Error(errorData.message || `Request failed with status ${response.status}`);
        }

        // Handle 204 No Content (DELETE)
        if (response.status === 204) {
            return null; // No body to parse
        }

        // Parse JSON for other successful responses
        return await response.json();

    } catch (error) {
        console.error('Network or API Call Error:', error);
        // Re-throw the error so the calling function can handle UI updates
        throw error;
    }
}

// --- UI Update Helpers ---
function showLoading(element, message = 'Loading...') {
    if (element) {
        element.innerHTML = `<li style="text-align: center; color: var(--text-secondary); padding: 20px;">${escapeHtml(message)}</li>`;
    }
}

function showError(element, message = 'Failed to load data.') {
    if (element) {
        element.innerHTML = `<li style="text-align: center; color: var(--error-color); padding: 20px;">${escapeHtml(message)}</li>`;
    }
}

// --- Placeholder Functions for Backend Interaction --- 

async function loadPlannerData() {
    console.log('Fetching planner data from API...');
    showLoading(taskListUl, 'Loading tasks...');
    showLoading(upcomingListUl, 'Loading upcoming...');
    showLoading(todayListUl, 'Loading today...');
    
    try {
        // Fetch all events - add query params later if needed for performance
        currentEvents = await apiCall(API_BASE_URL); 
        renderTaskList(currentEvents);
        renderWidgets(currentEvents); // Pass fetched events to widget renderer
    } catch (error) {
        console.error('Failed to load planner data:', error);
        showError(taskListUl, `Error loading tasks: ${error.message}`);
        showError(upcomingListUl, 'Error');
        showError(todayListUl, 'Error');
        currentEvents = []; // Clear events on error
    }
}

async function addPlannerEvent(eventData) {
    console.log('Adding event via API:', eventData);
    // Convert date/time to ISO UTC
    const startTimeIso = formatToISOUtc(eventData.date, eventData.time);
    // Assuming end_time is same as start_time for simple events, or requires another input
    // For simplicity, let's make end_time same as start_time. Adjust if needed.
    const endTimeIso = startTimeIso; 

    if (!startTimeIso) {
        alert('Invalid date or time provided.');
        return;
    }

    const apiPayload = {
        title: eventData.title,
        description: eventData.notes || null,
        start_time: startTimeIso,
        end_time: endTimeIso, // Adjust if separate end time is needed
        event_type: eventData.category, // Map form category to API's event_type
        is_completed: false // Default for new events
    };

    try {
        await apiCall(API_BASE_URL, 'POST', apiPayload);
        closeModal();
        loadPlannerData(); // Reload all data after successful add
    } catch (error) {
        alert(`Failed to add event: ${error.message}`);
    }
}

async function updatePlannerEvent(eventId, eventData) {
    console.log(`Updating event ${eventId} via API:`, eventData);
    const startTimeIso = formatToISOUtc(eventData.date, eventData.time);
    const endTimeIso = startTimeIso; // Adjust if separate end time is needed

    if (!startTimeIso) {
        alert('Invalid date or time provided.');
        return;
    }

    const apiPayload = {
        title: eventData.title,
        description: eventData.notes || null,
        start_time: startTimeIso,
        end_time: endTimeIso, // Adjust if needed
        event_type: eventData.category,
        // is_completed is handled by updateEventStatus
    };

    try {
        // Using PATCH for partial update, assuming backend supports it for all fields
        await apiCall(`${API_BASE_URL}${eventId}/`, 'PATCH', apiPayload);
        closeModal();
        loadPlannerData(); // Reload all data after successful update
    } catch (error) {
        alert(`Failed to update event: ${error.message}`);
    }
}

async function deletePlannerEvent(eventId) {
    console.log(`Deleting event ${eventId} via API.`);
    if (!confirm('Are you sure you want to delete this event?')) return;

    try {
        await apiCall(`${API_BASE_URL}${eventId}/`, 'DELETE');
        // Optimistic UI update (remove immediately)
        const listItem = taskListUl.querySelector(`li[data-event-id="${eventId}"]`);
        if (listItem) listItem.remove();
        // Remove from local state as well for widget consistency before full reload (optional)
        currentEvents = currentEvents.filter(event => event.id !== eventId);
        renderWidgets(currentEvents);
        // Consider a full reload if deletion affects other computed data: loadPlannerData(); 
    } catch (error) {
        alert(`Failed to delete event: ${error.message}`);
    }
}

async function updateEventStatus(eventId, isCompleted) {
    console.log(`Updating status for event ${eventId} to ${isCompleted} via API.`);
    const apiPayload = { is_completed: isCompleted };

    // Optimistic UI Update
    const listItem = taskListUl.querySelector(`li[data-event-id="${eventId}"]`);
    if (listItem) {
        listItem.classList.toggle('completed', isCompleted);
    }

    try {
        await apiCall(`${API_BASE_URL}${eventId}/`, 'PATCH', apiPayload);
        // Update local state for consistency
        const index = currentEvents.findIndex(event => event.id === eventId);
        if (index !== -1) {
            currentEvents[index].is_completed = isCompleted;
        }
        // No full reload needed, widgets can be updated if required based on completion status change
        renderWidgets(currentEvents); 
    } catch (error) {
        alert(`Failed to update status: ${error.message}`);
        // Revert optimistic UI update on error
        if (listItem) {
            listItem.classList.toggle('completed', !isCompleted);
        }
    }
}

// --- Video Suggestions (Placeholder) ---
async function getVideoSuggestions(topic) {
     console.log(`Placeholder: getVideoSuggestions('${topic}') called.`);
     // TODO: Implement API call to video suggestion endpoint when available
     // const suggestions = await apiCall(`/api/v1/video/suggestions?topic=${encodeURIComponent(topic)}`);
     alert(`Video suggestions requested for: ${topic}\n(API integration needed)`);
}

// --- UI Rendering Functions --- 

function renderTaskList(tasks) {
    if (!taskListUl) return; // Ensure DOM is ready
    taskListUl.innerHTML = ''; // Clear existing list
    if (!tasks || tasks.length === 0) {
        taskListUl.innerHTML = '<li style="text-align: center; color: var(--text-secondary); padding: 20px;">No tasks or events found.</li>';
        return;
    }

    // Sort tasks (e.g., by start_time, then incomplete first)
    const sortedTasks = [...tasks].sort((a, b) => {
        if (a.is_completed !== b.is_completed) return a.is_completed ? 1 : -1; // Incomplete first
        const dateA = new Date(a.start_time);
        const dateB = new Date(b.start_time);
        return dateA - dateB; // Then by start time ascending
    });

    sortedTasks.forEach(task => {
        const li = document.createElement('li');
        li.className = `task-item ${task.is_completed ? 'completed' : ''}`;
        li.dataset.eventId = task.id;

        const dateDisplay = formatDate(task.start_time);
        const timeDisplay = formatTime(task.start_time);
        // const endTimeDisplay = formatTime(task.end_time); // Optional: show end time too

        // Placeholder for video topic derived from title/description if needed
        const videoTopic = task.title; // Simple example, refine if needed

        li.innerHTML = `
            <input type="checkbox" class="task-checkbox" aria-label="Mark task ${task.is_completed ? 'incomplete' : 'complete'}" ${task.is_completed ? 'checked' : ''}>
            <div class="task-details">
                <span class="task-title">${escapeHtml(task.title)}</span>
                <div class="task-meta">
                    ${task.event_type ? `<span class="task-tag tag-${escapeHtml(task.event_type)}">${escapeHtml(task.event_type.charAt(0).toUpperCase() + task.event_type.slice(1))}</span>` : ''}
                    <span class="due-date"><i class="far fa-clock"></i> ${dateDisplay}${timeDisplay ? ', ' + timeDisplay : ''}</span>
                </div>
                ${videoTopic ? 
                    `<div class="video-suggestion-placeholder" data-topic="${escapeHtml(videoTopic)}">
                        <i class="fas fa-video"></i> Video suggestions for '${escapeHtml(videoTopic)}'
                    </div>` : ''}
            </div>
            <div class="task-actions">
                <button class="btn-icon edit-btn" title="Edit"><i class="fas fa-pencil-alt"></i></button>
                <button class="btn-icon delete-btn" title="Delete"><i class="fas fa-trash-alt"></i></button>
            </div>
        `;

        // Add event listeners for this task item
        li.querySelector('.task-checkbox').addEventListener('change', (e) => {
            updateEventStatus(task.id, e.target.checked);
        });
        li.querySelector('.edit-btn').addEventListener('click', () => {
            openModalForEdit(task.id);
        });
        li.querySelector('.delete-btn').addEventListener('click', () => {
            deletePlannerEvent(task.id);
        });
        const videoPlaceholder = li.querySelector('.video-suggestion-placeholder');
        if (videoPlaceholder) {
            videoPlaceholder.addEventListener('click', () => {
                getVideoSuggestions(videoPlaceholder.dataset.topic);
            });
        }

        taskListUl.appendChild(li);
    });
}

// Modify renderWidgets to process the full event list
function renderWidgets(allEvents) {
    if (!upcomingListUl || !todayListUl) return; // Ensure DOM is ready
    
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const todayEnd = new Date(todayStart.getTime() + 24 * 60 * 60 * 1000);
    const tomorrowStart = new Date(todayEnd);
    const tomorrowEnd = new Date(tomorrowStart.getTime() + 24 * 60 * 60 * 1000);
    const next7DaysEnd = new Date(todayStart.getTime() + 7 * 24 * 60 * 60 * 1000);

    // Filter for today's events (start_time is within today)
    const todayEvents = allEvents
        .filter(event => {
            const eventStart = new Date(event.start_time);
            return eventStart >= todayStart && eventStart < todayEnd;
        })
        .sort((a, b) => new Date(a.start_time) - new Date(b.start_time)); // Sort by time

    // Filter for upcoming events (not completed, start_time is after now, within next 7 days)
     const upcomingEvents = allEvents
        .filter(event => {
            const eventStart = new Date(event.start_time);
            return !event.is_completed && eventStart >= now && eventStart < next7DaysEnd;
         })
        .sort((a, b) => new Date(a.start_time) - new Date(b.start_time))
        .slice(0, 5); // Limit to top 5 upcoming

    // Render Upcoming List
    upcomingListUl.innerHTML = '';
    if (upcomingEvents.length > 0) {
        upcomingEvents.forEach(item => {
            const li = document.createElement('li');
            li.className = 'widget-list-item';
            li.innerHTML = `
                <span class="item-title">${escapeHtml(item.title)}</span>
                <span class="item-date">${formatDate(item.start_time)}</span>
            `;
            upcomingListUl.appendChild(li);
        });
    } else {
        upcomingListUl.innerHTML = '<li style="color: var(--text-secondary); font-size: 0.9rem;">Nothing upcoming soon.</li>';
    }

    // Render Today List
    todayListUl.innerHTML = '';
    if (todayEvents.length > 0) {
        todayEvents.forEach(item => {
            const li = document.createElement('li');
            li.className = 'widget-list-item';
            li.innerHTML = `
                <span class="item-title">${escapeHtml(item.title)}</span>
                <span class="item-time">${formatTime(item.start_time)}</span>
            `;
            todayListUl.appendChild(li);
        });
    } else {
        todayListUl.innerHTML = '<li style="color: var(--text-secondary); font-size: 0.9rem;">Nothing scheduled for today.</li>';
    }
}

// --- Modal Handling --- 

function openModal() {
    if (!eventForm) return;
    eventForm.reset(); // Clear form
    eventIdInput.value = ''; // Clear hidden ID
    modalTitle.textContent = 'Add New Event';
    eventModal.classList.add('show'); // Use class for animation
}

function openModalForEdit(eventId) {
    if (!eventForm) return;
    // Find the event in the locally stored data
    const task = currentEvents.find(t => t.id == eventId);
    if (!task) {
        alert('Event not found.');
        return;
    }

    modalTitle.textContent = 'Edit Event';
    eventIdInput.value = task.id;
    eventTitleInput.value = task.title;
    eventTypeInput.value = task.event_type; // Use event_type from API
    eventCourseInput.value = task.event_type; // Assuming category maps directly to event_type
    eventDateInput.value = formatIsoToDateInput(task.start_time);
    eventTimeInput.value = formatIsoToTimeInput(task.start_time);
    eventNotesInput.value = task.description || '';

    eventModal.classList.add('show');
}

function closeModal() {
    if (eventModal) eventModal.classList.remove('show');
}

// Utility to escape HTML to prevent XSS
function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') return unsafe;
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
 }

// --- Event Listeners Setup --- 
function initializePlanner() {
    // Cache DOM elements
    taskListUl = document.getElementById('taskList');
    upcomingListUl = document.getElementById('upcomingList');
    todayListUl = document.getElementById('todayList');
    addEventBtn = document.getElementById('addEventBtn');
    eventModal = document.getElementById('eventModal');
    modalCloseBtn = document.getElementById('modalCloseBtn');
    modalCancelBtn = document.getElementById('modalCancelBtn');
    eventForm = document.getElementById('eventForm');
    modalTitle = document.getElementById('modalTitle');
    eventIdInput = document.getElementById('eventId');
    eventTitleInput = document.getElementById('eventTitle');
    eventTypeInput = document.getElementById('eventType'); // Note: this selects based on type (assignment, deadline etc)
    eventCourseInput = document.getElementById('eventCourse'); // This selects based on course (calculus, physics etc)
    eventDateInput = document.getElementById('eventDate');
    eventTimeInput = document.getElementById('eventTime');
    eventNotesInput = document.getElementById('eventNotes');

    // Attach global listeners
    if (addEventBtn) addEventBtn.addEventListener('click', openModal);
    if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeModal);
    if (modalCancelBtn) modalCancelBtn.addEventListener('click', closeModal);
    if (eventModal) {
        eventModal.addEventListener('click', (e) => { // Close modal if backdrop is clicked
            if (e.target === eventModal) {
                closeModal();
            }
        });
    }

    if (eventForm) {
        eventForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const formData = new FormData(eventForm);
            // Get data from form for add/edit payload
            const eventData = {
                // Use form field names directly
                title: formData.get('title') || '', 
                // Map form fields to API expectations
                category: formData.get('category') || 'general', // Use 'category' from form for event_type API field 
                date: formData.get('date'), 
                time: formData.get('time') || null, 
                notes: formData.get('notes') || '',
            };

            if (!eventData.title || !eventData.date) {
                alert('Please provide at least a title and date.');
                return;
            }

            const eventId = eventIdInput.value;
            if (eventId) {
                // Editing existing event
                updatePlannerEvent(eventId, eventData);
            } else {
                // Adding new event
                addPlannerEvent(eventData);
            }
        });
    }

    // --- Initial Load ---
    // Ensure DOM is fully loaded before trying to access elements and fetch data
    loadPlannerData();
}

// --- Initialize ---
document.addEventListener('DOMContentLoaded', initializePlanner);
