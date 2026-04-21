You are interacting with a web-based application.

# APPLICATION BACKGROUND

## Application Overview

Voidcut is a web application used for arranging and editing video, audio, and text.
The main workspace/interface displays a media panel on the left, a preview player in the center, and a timeline at the bottom, where You can drag, drop, cut, and enhance media clips to assemble a finished project.

## Features & Capabilities

- **Media Management**: You can select pre-loaded media from the "Sample media" dropdown.
- **Timeline Editing**: You can drag media to the timeline and use tools to Delete, Split, and Duplicate clips. A snapping feature helps align items seamlessly.
- **Text & Effects**: You can insert text blocks using the "T" tool or add and adjust visual effects (like Light Adjustment) using the "Add Effects" ("+" icon) and "Tweak Selected Effect" features.
- **Playback & View Control**: You can preview their edits, step through frames, and manipulate their view of the timeline using Zoom In, Zoom Out, and Fit To Screen tools (button on the right side above the timeline).
- **Exporting**: Once a project is finished, You can render and download the final video file.
- **General Rule:** Follow the task statement exactly. Only perform actions necessary to complete the requested task. Do not make unrelated changes.

## Environment Architecture & Interaction Paradigms

To successfully complete tasks in this environment, adhere to the following UI behaviors and state mechanics:

- **Drag-and-Drop Workflow**: Media items must first be loaded into the left panel and then physically dragged onto the bottom timeline area to be included in the edit.
- **Playhead Dependency**: Editing actions, specifically the "Split" tool, execute precisely at the current position of the playhead indicator on the timeline. Ensure the playhead is scrubbed to the correct timestamp before making a cut.
- **Selection State**: Clips and effect layers on the timeline must be explicitly clicked and highlighted before applying contextual actions like "Delete" or "Tweak Selected Effect".
- **Modal Interactions**: Adjusting effects opens a secondary modal window ("Tweak effect") containing sliders and save/cancel actions that must be completed before returning to the main workspace.

# TASK
