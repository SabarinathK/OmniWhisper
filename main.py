import asyncio
import os
import sys

import sounddevice as sd
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.audio.vad_processor import VADProcessor

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.pipeline.runner import WorkerRunner

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.tts import GroqTTSService
from pipecat.services.openrouter.llm import OpenRouterLLMService

from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO")

# Your microphone and speaker
sd.default.device = (1, 4)


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=1,
            output_device_index=4,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_in_stream_on_start=True,
        )
    )

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    stt = GroqSTTService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqSTTService.Settings(
            model="whisper-large-v3-turbo",
        ),
    )

    llm = OpenRouterLLMService(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        settings=OpenRouterLLMService.Settings(
            model="openrouter/free",
            temperature=0.7,
            max_completion_tokens=200,
        ),
    )

    tts = GroqTTSService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqTTSService.Settings(
            voice="autumn",
        ),
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are OmniWhisper, a helpful AI voice assistant. "
                "Keep responses concise, natural, and conversational. "
                "Avoid markdown, bullet points, code blocks, and emojis. "
                "Respond as if speaking to a human."
            ),
        }
    ]

    context = LLMContext(messages)

    user_context, assistant_context = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            user_context,
            llm,
            tts,
            transport.output(),
            assistant_context,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        name="OmniWhisper",
    )

    runner = WorkerRunner()

    await runner.add_workers(worker)

    logger.info("🎤 OmniWhisper started")
    logger.info("Speak into your microphone...")

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
