You are interacting with a web-based application.

# APPLICATION BACKGROUND

## Application Overview

CircuitJS is a web application used for designing and simulating analog and digital circuits.  
The main workspace displays a circuit canvas, where users can place components, wire them together, and observe live simulation behavior.

## Features & Capabilities

- **Component Placement:** Add parts from the **Draw** menu, top search button, or by right-clicking empty canvas space.
- **Circuit Wiring:** Connect terminals by clicking and dragging between component endpoints. Crossing wires do not connect unless joined at endpoints.
- **Component Editing:** Right-click a placed component to edit values, duplicate, or delete it.
- **Measurement:** Hover over a component to view live values in the bottom-right information panel.
- **General Rule:** Follow the task statement exactly. Only perform actions necessary to complete the requested task. Do not make unrelated changes.

## Environment Architecture & Interaction Paradigms

To successfully complete tasks in this environment, adhere to the following UI behaviors and state mechanics:

- **Right-Click Menus:** Most components are added through nested right-click menus on empty canvas space.
- **Property Dialogs:** Values such as resistance or voltage are changed through popup dialogs after selecting **Edit**.
- **Drag Wiring:** Connections are made by dragging from one terminal to another.
- **Logic Input Placement:** Draw logic inputs from right to left for proper orientation.
- **Live Updates:** Simulation runs continuously, so allow time for values to stabilize before measuring.

# TASK