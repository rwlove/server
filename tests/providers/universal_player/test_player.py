"""Regression tests for universal player command proxying (#5443)."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import PlaybackState, PlayerFeature, PlayerType

from music_assistant.models.player import DeviceInfo, Player
from music_assistant.providers.universal_player.player import UniversalPlayer
from music_assistant.providers.universal_player.provider import UniversalPlayerProvider


def _make_mock_mass() -> MagicMock:
    mass = MagicMock()
    mass.closing = False
    mass.config = MagicMock()
    mass.config.get = MagicMock(return_value=[])

    def _get_raw_player_config_value(
        _player_id: str, key: str, default: object | None = None
    ) -> object | None:
        if key == "min_volume":
            return 0
        if key == "max_volume":
            return 100
        return default if default is not None else "auto"

    mass.config.get_raw_player_config_value = MagicMock(side_effect=_get_raw_player_config_value)
    mass.config.get_raw_core_config_value = MagicMock(return_value="GLOBAL")
    mass.config.set = MagicMock()
    mass.signal_event = MagicMock()
    mass.get_providers = MagicMock(return_value=[])
    return mass


def _make_universal_provider(mock_mass: MagicMock) -> UniversalPlayerProvider:
    manifest = MagicMock()
    manifest.domain = "universal_player"
    manifest.name = "Universal Player"
    provider = UniversalPlayerProvider.__new__(UniversalPlayerProvider)
    provider.mass = mock_mass
    provider.manifest = manifest
    provider.logger = logging.getLogger("test.universal_player")
    config = MagicMock()
    config.instance_id = "universal_player"
    config.name = None
    provider.config = config
    provider._universal_player_locks = {}
    return provider


def _make_chromecast_player(
    mass: MagicMock,
    player_id: str,
    *,
    active_source: str | None,
    features: set[PlayerFeature],
) -> MagicMock:
    """Return a chromecast-domain protocol player with the given active source and features."""
    player = MagicMock(spec=Player)
    player.player_id = player_id
    player.available = True
    player.active_source = active_source
    player.playback_state = PlaybackState.PLAYING
    player.supported_features = features
    player.volume_level = 42
    player.volume_muted = True
    player.powered = True
    provider = MagicMock()
    provider.domain = "chromecast"
    player.provider = provider
    player.volume_set = AsyncMock()
    player.volume_mute = AsyncMock()
    player.power = AsyncMock()
    player.play = AsyncMock()
    player.pause = AsyncMock()
    player.stop = AsyncMock()
    mass.players.get_player = MagicMock(return_value=player)
    return player


def _make_universal_player(mass: MagicMock, protocol_player_ids: list[str]) -> UniversalPlayer:
    provider = _make_universal_provider(mass)
    base_cfg = MagicMock()
    base_cfg.name = None
    base_cfg.default_name = "Universal"
    mass.config.get_base_player_config.return_value = base_cfg
    player = UniversalPlayer(
        provider=provider,
        player_id="up_test",
        name="Universal",
        device_info=DeviceInfo(model="Universal Player", manufacturer="Music Assistant"),
        protocol_player_ids=list(protocol_player_ids),
    )
    player._attr_available = True
    player._attr_type = PlayerType.PLAYER
    player._cache.clear()
    player.set_initialized()
    return player


@pytest.fixture
def setup() -> tuple[UniversalPlayer, MagicMock]:
    """Universal player with a chromecast running Spotify Connect."""
    mass = _make_mock_mass()
    chromecast = _make_chromecast_player(
        mass,
        "cc_1",
        active_source="spotify_connect",
        features={
            PlayerFeature.VOLUME_SET,
            PlayerFeature.VOLUME_MUTE,
            PlayerFeature.POWER,
            PlayerFeature.PLAY_MEDIA,
        },
    )
    universal = _make_universal_player(mass, ["cc_1"])
    return universal, chromecast


async def test_volume_set_proxied_to_external_source(
    setup: tuple[UniversalPlayer, MagicMock],
) -> None:
    """volume_set on the universal player delegates to the active external source."""
    universal, chromecast = setup
    await universal.volume_set(42)
    chromecast.volume_set.assert_awaited_once_with(42)


async def test_volume_mute_proxied_to_external_source(
    setup: tuple[UniversalPlayer, MagicMock],
) -> None:
    """volume_mute on the universal player delegates to the active external source."""
    universal, chromecast = setup
    await universal.volume_mute(True)
    chromecast.volume_mute.assert_awaited_once_with(True)


async def test_power_proxied_to_external_source(
    setup: tuple[UniversalPlayer, MagicMock],
) -> None:
    """Power on the universal player delegates to the active external source."""
    universal, chromecast = setup
    await universal.power(False)
    chromecast.power.assert_awaited_once_with(False)


def test_supported_features_excludes_play_media(
    setup: tuple[UniversalPlayer, MagicMock],
) -> None:
    """Universal player never advertises PLAY_MEDIA even when the external source does."""
    universal, _chromecast = setup
    features = universal.supported_features
    assert PlayerFeature.PLAY_MEDIA not in features
    assert PlayerFeature.VOLUME_SET in features
    assert PlayerFeature.POWER in features


def test_state_read_back_from_external_source(
    setup: tuple[UniversalPlayer, MagicMock],
) -> None:
    """Volume/mute/power are reported from the active external source."""
    universal, chromecast = setup
    assert universal.volume_level == chromecast.volume_level
    assert universal.volume_muted == chromecast.volume_muted
    assert universal.powered == chromecast.powered


def test_state_falls_back_to_base_attributes_without_external_source() -> None:
    """Without an active external source, volume/mute/power use the player's own state."""
    mass = _make_mock_mass()
    _make_chromecast_player(
        mass,
        "cc_1",
        active_source=None,
        features={PlayerFeature.VOLUME_SET, PlayerFeature.VOLUME_MUTE, PlayerFeature.POWER},
    )
    universal = _make_universal_player(mass, ["cc_1"])
    universal._attr_volume_level = 13
    universal._attr_volume_muted = False
    universal._attr_powered = False
    assert universal.volume_level == 13
    assert universal.volume_muted is False
    assert universal.powered is False
