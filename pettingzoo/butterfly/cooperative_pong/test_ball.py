"""Test that Ball behaves correctly."""

import math

import pygame
import pytest
from gymnasium.utils import seeding

from pettingzoo.butterfly.cooperative_pong.ball import Ball
from pettingzoo.butterfly.cooperative_pong.cooperative_pong import get_valid_angle
from pettingzoo.butterfly.cooperative_pong.paddle import Paddle


def test_bounds() -> None:
    """Run multiple ball trajectories to confirm bounce works correctly."""
    # make ball size, screen size, ball speed match env default
    render_ratio = 2
    width, height = 960 // render_ratio, 560 // render_ratio
    ball_dims = (20 // render_ratio, 20 // render_ratio)
    ball_speed = 9

    area = pygame.Rect((0, 0, width, height))
    p0 = Paddle((4, 4), 0, "left")
    p1 = Paddle((4, 4), 0, "right")

    n_tests = 10000
    for seed in range(n_tests):
        randomizer, seed = seeding.np_random(seed=seed)
        ball = Ball(randomizer, ball_dims, ball_speed, bounce_randomness=False)

        ball.reset(area.center, get_valid_angle(randomizer))

        while True:
            ball.update2(area, p0, p1)
            terminate = ball.is_out_of_bounds()

            in_bounds = ball._rect.x >= 0 and ball._rect.x <= width - ball_dims[1]
            if not terminate and not in_bounds:
                raise AssertionError(
                    f"Cooperative pong not terminated with ball out of bounds (seed = {seed})"
                )
            if terminate:
                break


def test_bounce_randomness() -> None:
    """Confirm that the speed remains constant with bounce randomness enabled."""

    def calc_speed(ball: Ball) -> float:
        return math.sqrt(ball._speed[0] ** 2 + ball._speed[1] ** 2)

    # make ball size, screen size, ball speed match env default
    render_ratio = 2
    width, height = 960 // render_ratio, 560 // render_ratio
    ball_dims = (20 // render_ratio, 20 // render_ratio)
    ball_speed = 9

    area = pygame.Rect((0, 0, width, height))

    # paddles are tall enough to cover entire side
    # and wide enough that the ball can't move through them
    p0 = Paddle((ball_speed + 5, height), 0, "left")
    p1 = Paddle((ball_speed + 5, height), 0, "right")
    p0.rects[0].midleft = area.midleft
    p1.rects[0].midright = area.midright

    randomizer, _ = seeding.np_random(seed=11)

    ball = Ball(randomizer, ball_dims, ball_speed, bounce_randomness=True)
    ball.reset(area.center, get_valid_angle(randomizer))

    initial_speed = pytest.approx(calc_speed(ball))
    for _ in range(10000):
        ball.update2(area, p0, p1)
        terminate = ball.is_out_of_bounds()

        current_speed = calc_speed(ball)
        assert initial_speed == current_speed, "Ball speed changed unexpectedly"

        if terminate:
            # the paddles occupy the entire side walls, so it should not be
            # possible for a ball to pass by. So a termination should never
            # happen if everything is working properly. Things to check:
            # 1) did the height of the paddles above get changed to not be
            #    the full height of the screen?
            # 2) did the width of the paddles get changed to be less than
            #    the total speed (plus a small buffer)?
            # 3) did something get added to limit the number of steps?
            #    If so, check that it impacts this test code and that the
            #    number of steps run here is less than than limit.
            # 4) something is wrong with the paddle/ball to allow the ball to
            #    bypass a paddle or terminate without the ball hitting a wall.
            raise AssertionError("Unknown error in test (seed test code)")


@pytest.mark.parametrize("ball_speed", [3, 9, 20, 40])
@pytest.mark.parametrize("wall", ["top", "bottom"])
def test_wall_bounce_is_reflected_not_clamped(ball_speed: int, wall: str) -> None:
    """A ball crossing a horizontal wall is mirrored back across it.

    Resting the ball against the wall instead would discard the distance it
    travelled after the crossing, and that distance grows with speed, so the
    error would be invisible at low speed and large at high speed.
    """
    render_ratio = 2
    width, height = 960 // render_ratio, 560 // render_ratio
    ball_dims = (20 // render_ratio, 20 // render_ratio)

    area = pygame.Rect((0, 0, width, height))
    # keep the paddles off screen so only the wall is under test
    p0 = Paddle((4, 4), 0, "left")
    p1 = Paddle((4, 4), 0, "right")
    p0.rects[0].topleft = (-100, -100)
    p1.rects[0].topleft = (width + 100, -100)

    randomizer, _ = seeding.np_random(seed=0)
    ball = Ball(randomizer, ball_dims, ball_speed, bounce_randomness=False)

    # aim straight at the wall, starting one pixel away from it
    ball.reset(area.center, math.pi / 2 if wall == "bottom" else -math.pi / 2)
    if wall == "bottom":
        ball._rect.bottom = area.bottom - 1
        edge_before = ball._rect.bottom
    else:
        ball._rect.top = area.top + 1
        edge_before = ball._rect.top

    speed_before = ball._speed[1]
    ball.update2(area, p0, p1)

    # where the ball would have been with no wall, mirrored back across it
    wall_pos = area.bottom if wall == "bottom" else area.top
    expected = 2 * wall_pos - (edge_before + speed_before)
    edge_after = ball._rect.bottom if wall == "bottom" else ball._rect.top

    assert edge_after == pytest.approx(expected, abs=1), (
        f"Ball was placed at {edge_after} instead of the mirrored position "
        f"{expected} after crossing the {wall} wall at speed {ball_speed}"
    )
    assert ball._speed[1] == pytest.approx(-speed_before), (
        "Vertical velocity should reverse on a wall bounce"
    )
    assert not ball.is_out_of_bounds(), (
        "A ball bouncing off a horizontal wall is not out of bounds"
    )


if __name__ == "__main__":
    test_bounce_randomness()
