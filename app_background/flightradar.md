You are interacting with a web-based application.

# APPLICATION BACKGROUND

## Application Overview

Flight Radar is a web application used for tracking, visualizing, and replaying aircraft flight data.
The main workspace/interface displays a map populated with aircraft icons alongside a task panel, where users can search for specific flights, adjust historical playback times, view flight trajectories, and extract detailed telemetry.

## Features & Capabilities

- **Temporal Navigation**: Adjust the temporal state of the map using the playback controls and timeline slider located at the top center of the screen to set exact UTC times.
- **Flight Search**: Locate specific aircraft by typing their callsign into the Search bar situated in the top left corner.
- **Flight Details View**: Selecting an aircraft highlights its flight path and opens a detailed panel overlaying the map, detailing its route, origin/destination, current status, and specific Flight Data (Altitude, Geo Altitude, Ground Speed, Vertical Rate).
- **Radius Tool**: Draw or clear dashed boundaries around target airports using the "Draw Radius Circle" tool (accessed via the target icon next to the pause button).
- **General Rule:** Follow the task statement exactly. Only perform actions necessary to complete the requested task. Do not make unrelated changes.

## Environment Architecture & Interaction Paradigms

To successfully complete tasks in this environment, adhere to the following UI behaviors and state mechanics:

- **Sequential Data Retrieval**: The interface displays detailed data for one flight at a time.
- **Panel Scrolling**: Specific telemetry metrics are located under the "Flight Data" section of the details panel. You may need to scroll down within the panel to locate fields like Geo Altitude or Ground Speed.

# TASK