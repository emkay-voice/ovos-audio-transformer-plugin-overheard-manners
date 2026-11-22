"""Stage 2's ears: speech to text that runs on this device, never on a server.

What the room says must not leave the device, so the plugin only uses an STT
engine known to run locally. It is either named in this plugin's own config
(``"stt": {"module": ..., ...}``) or, if not, the listener's configured STT, but
only when that one is on the list below. With a server STT (OpenVoiceOS'
public servers, a cloud service) and nothing local configured, the plugin turns
itself off and says why in the log.
"""
from typing import Optional

from ovos_utils.log import LOG

#: STT plugins that transcribe on the device itself. A plugin missing here can
#: still be used by naming it in this plugin's "stt" config, which is taken as
#: the user's statement that it is local.
LOCAL_STT = {
    "ovos-stt-plugin-fasterwhisper",
    "ovos-stt-plugin-whispercpp",
    "ovos-stt-plugin-whisper",
    "ovos-stt-plugin-vosk",
    "ovos-stt-plugin-vosk-streaming",
    "ovos-stt-plugin-citrinet",
    "ovos-stt-plugin-nemo",
    "ovos-stt-plugin-wav2vec",
    "ovos-stt-plugin-sherpa-onnx",
    "ovos-stt-plugin-moonshine",
    "ovos-stt-plugin-parakeet",
    "ovos-stt-plugin-pocketsphinx",
}


def pick_local_stt(own: Optional[dict], listener_stt: Optional[dict]) -> Optional[dict]:
    """The STT config to use, or None if only a server STT is available."""
    if own and own.get("module"):
        return own
    listener_stt = listener_stt or {}
    module = listener_stt.get("module")
    if module in LOCAL_STT:
        return {"module": module, **(listener_stt.get(module) or {})}
    fallback = listener_stt.get("fallback_module")
    if fallback in LOCAL_STT:
        return {"module": fallback, **(listener_stt.get(fallback) or {})}
    return None


def load_stt(config: dict):
    """Load the STT plugin named in ``config``."""
    from ovos_plugin_manager.stt import load_stt_plugin
    clazz = load_stt_plugin(config["module"])
    if clazz is None:
        raise RuntimeError(f"STT plugin {config['module']} is not installed")
    plugin_config = {k: v for k, v in config.items() if k != "module"}
    LOG.info(f"overheard manners: transcribing on the device with {config['module']}")
    return clazz(plugin_config)


def transcribe(stt, pcm16: bytes, lang: str, sample_rate: int = 16000) -> str:
    from ovos_plugin_manager.utils.audio import AudioData
    return stt.execute(AudioData(pcm16, sample_rate, 2), language=lang) or ""
