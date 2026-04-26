# Dograh: Qwen-TTS Local Integration Report

This report documents the architectural changes made to the Dograh platform to support local `Qwen-TTS` voice cloning with WebSocket streaming.

---

## 1. Backend Integration (Pipecat)

### `pipecat/src/pipecat/services/qwen_tts.py`
- **Purpose**: Implements a dedicated Pipecat service for communicating with a local Qwen3-TTS WebSocket server.
- **Key Features**:
    - Inherits from `WebsocketTTSService`.
    - Handles WebSocket handshakes (`connect`, `config`, `synthesize`, `close_context`).
    - Low-latency streaming of raw PCM audio chunks.
    - Context management for handling multiple simultaneous synthesis requests.

---

## 2. API Layer Integration

### `api/services/configuration/registry.py`
- **Provider Definition**: Added `QWENTTS = "qwentts"` to the `ServiceProviders` Enum.
- **Configuration Class**: Defined `QwenTTSConfiguration` inheriting from `BaseTTSConfiguration`.
    - Fields: `api_url` (Custom endpoint), `voice` (Cloned ID), `model`, `api_key`.
- **Registry Update**: Added `QwenTTSConfiguration` to the `TTSConfig` Union for type validation.

### `api/services/pipecat/service_factory.py`
- **Factory Logic**: Updated `create_tts_service` to recognize the `QWENTTS` provider.
- **Initialization**: Instantiates the `QwenTTSService` using the `api_url`, `voice`, and `api_key` provided in the user's configuration.

### `api/routes/user.py`
- **Type Literals**: Added `"qwentts"` to the `TTSProvider` Literal definition.
- **Voice Discovery**: Implemented a static fallback for the `/configurations/voices/qwentts` endpoint to return a placeholder "Custom Cloned Voice" for selection in the UI.

---

## 3. Frontend Integration (UI)

### `ui/src/components/ServiceConfigurationForm.tsx`
- **Provider Visibility**: Added "QwenTTS" to the provider selection list.
- **Custom UI Fields**: Implemented conditional rendering for `qwentts`, surfacing specific inputs for:
    - **API URL**: The WebSocket endpoint of the cloning server.
    - **Voice ID**: The manual ID of the cloned persona.

### `ui/src/components/VoiceSelector.tsx`
- **Provider Mapping**: Added `qwentts` to the `TTSProviderWithVoices` type to ensure UI compatibility.

---

## 4. Maintenance & Deployment
- The integration is designed to be "Docker-ready," allowing local services to communicate over the internal bridge or host network.
- Configuration is persisted in the Dograh database via the standard user settings workflow.
